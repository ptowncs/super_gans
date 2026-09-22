import gc
import os

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from pytorch_fid import fid_score

#from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import vgg19
from torchvision.utils import *
from tqdm import tqdm

import super_gans.config as cfg
import wandb
from super_gans import utils
from super_gans import metrics

# Force cuDNN to use a deterministic algorithm instead of searching
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
# Turn off cuDNN entirely for convolutions to get around kaggle conflicts
torch.backends.cudnn.enabled = False

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        # Code Fix: Enforcing bias=False when use_bn=True is handled safely here
        self.cnn = nn.Conv2d(in_channels, out_channels, bias=not use_bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True) if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        # Code Fix: Clean sequential flow matching your conditional activation parameters
        if self.use_act:
            return self.act(self.bn(self.cnn(x)))
        return self.bn(self.cnn(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor=2):
        super().__init__()
        # Correctly tracks channel transformations: in_c -> in_c * 4 -> PixelShuffle -> in_c
        self.conv = nn.Conv2d(in_c, in_c * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)

    def forward(self, x):
        return self.block2(self.block1(x)) + x

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, use_act=False, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))

        # Code Fix: Disabled internal convolutional bias here to protect Tanh from early saturation collapse
        self.final = nn.Conv2d(num_channels, out_channels, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, x):
        initial = self.initial(x)
        out = self.residuals(initial)
        out = self.convblock(out) + initial
        out = self.upsamples(out)
        return torch.tanh(self.final(out))

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        current_in_channels = in_channels

        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels=current_in_channels,
                    out_channels=feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            current_in_channels = feature

        self.blocks = nn.Sequential(*blocks)

        # Code Fix: Linear layer input size is now dynamically scaled via 'current_in_channels'
        # to guarantee the model never crashes if you alter your configuration feature arrays.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(current_in_channels * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.classifier(self.blocks(x))

# phi_5,4 5th conv layer before maxpooling but after activation
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(cfg.device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

def train_fn(epoch, loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb, start_epoch):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(cfg.device)
        low_res = low_res.to(cfg.device)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))

        # --- FIXED FOR 1-CHANNEL DATA TO 3-CHANNEL VGG LOSS ---
        # Stack the 1-channel tensor 3 times along the channel dimension (dim=1)
        # This turns [Batch, 1, 128, 128] -> [Batch, 3, 128, 128]
        fake_rgb = torch.cat([fake, fake, fake], dim=1)
        high_res_rgb = torch.cat([high_res, high_res, high_res], dim=1)
        loss_for_vgg = 0.006 * vgg_loss(fake_rgb, high_res_rgb)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()
        # --- LOG LOSSES EVERY EPOCH ---
        # Staging loss data. commit=False ensures we wait to push until the end of the epoch.
        wandb.log({"Loss/Discriminator": loss_disc.item(), "Loss/Generator": gen_loss.item(), "epoch": epoch}, commit=False)

        # --- VISUALS AT START OF EPOCH ---
        if idx == 0 and (epoch == start_epoch or epoch % 10 == 0):
            print(f"Epoch [{epoch}/{cfg.num_epochs}] Loss D: {loss_disc.item():.4f}, Loss G: {gen_loss.item():.4f}")
            utils.log_tensorboard_visuals(wandb, gen, high_res, low_res, epoch)


def training_loop(loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb, start_epoch=1, best_fid=float('inf'), best_fid_epoch=0):
    # feature=64 uses a lower layer of Inception; it's faster for monitoring
    fid_metric = FrechetInceptionDistance(feature=cfg.fid_dims, normalize=True).to(cfg.device)

    # Define master timeline metric
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")

    # Setup CSV logging
    train_csv_path = utils.setup_training_csv("srgan")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        process = psutil.Process(os.getpid())
        print(f"Epoch: {epoch} | RAM GB: {process.memory_info().rss / 1024**3:.2f} \
              | GPU GB: {torch.cuda.memory_allocated() / 1024**3:.2f}")
        train_fn(epoch, loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb)


        # --- FID CALCULATION AT END OF EPOCH ---
        if (epoch == start_epoch) or (epoch % cfg.fid_interval == 0) or (epoch == cfg.num_epochs):
            utils.save_model(gen, disc, opt_gen, opt_disc, epoch, filename="latest_gan.pth")
            # Swap loader to yield (high_res, low_res) for FID calculation: (real, generator_input)
            current_fid = metrics.compute_fid_from_real_and_input(gen, ((high_res, low_res) for low_res, high_res in loader), fid_metric)
            fid_metric.reset()

            wandb.log({"Metrics/FID": current_fid, "epoch": epoch}, commit=True)
            print(f"--- Epoch [{epoch}] FID Score: {current_fid:.4f} ---")

            # Log losses and FID to training CSV
            utils.log_training_row(train_csv_path, epoch,
                                  loss_g=gen_loss.item(),
                                  loss_d=loss_disc.item(),
                                  fid_train=current_fid)
            print(f"Epoch [{epoch}/{cfg.num_epochs}] CSV: G_loss={gen_loss.item():.6f}, D_loss={loss_disc.item():.6f}, FID_train={current_fid:.6f}")

            # Checkpoint: Save as 'best' if quality improved
            if current_fid < best_fid:
                best_fid = current_fid
                best_fid_epoch = epoch
                utils.save_model(gen, disc, opt_gen, opt_disc, epoch, filename="best_gan.pth")
            print(f"Epoch [{epoch}/{cfg.num_epochs}]: Best_fid_score={best_fid:.4f}, Best_fid_epoch={best_fid_epoch}")
        else:
            wandb.log({"epoch": epoch}, commit=True)
            print(f"Epoch [{epoch}/{cfg.num_epochs}] CSV: G_loss={gen_loss.item():.6f}, D_loss={loss_disc.item():.6f}, FID_train=N/A")

        # End of Epoch cleanup
        #writer.flush()
        torch.cuda.empty_cache()
        gc.collect()

    # --- FINAL RUN SUMMARY ---
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid

    return opt_disc, opt_gen, best_fid, best_fid_epoch

def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("sr-gan-model", type="model")
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    file_path = f"{gan_checkpoints_dir}/best_gan.pth"
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def createWandB():
    wandb.init(
        project="super-gans-project",
        name="SR_GAN",
        config={
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "z_dim": cfg.z_dim,
            "image_size": cfg.image_size,
            "num_channels": cfg.num_channels,
        },
    )
    return wandb

def load_datapairs(split="train"):
    # Load the base dataset using the standard data loading mechanism
    # This handles both Kaggle and RSNA datasets correctly
    base_dataset = utils.load_data(split=split)

    class SRGANDatasetWrapper(Dataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            # Get the image from the base dataset (returns image, label)
            # We ignore the label for SR-GAN training
            image_tensor, _ = self.dataset[index]

            # Convert tensor to numpy array in [0, 255] range for albumentations
            # The image tensor is in shape (C, H, W) with values in [-1, 1]
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = ((image_np + 1.0) * 127.5).astype(np.uint8)

            # Apply transforms to get low-res and high-res versions
            # both_transforms includes resizing, horizontal flip, affine transform
            transformed = cfg.both_transforms(image=image_np)
            transformed_image = transformed["image"]

            # Get high-res and low-res versions
            high_res = cfg.high_res_transform(image=transformed_image)["image"]
            low_res = cfg.low_res_transform(image=transformed_image)["image"]

            return low_res, high_res

    return SRGANDatasetWrapper(base_dataset)

# Wrapper class to adapt SRGAN dataset for saving real images (expects (image, label) format)
class SRGANRealImageWrapper:
    def __init__(self, sr_dataset):
        self.sr_dataset = sr_dataset

    def __len__(self):
        return len(self.sr_dataset)

    def __getitem__(self, idx):
        # SRGAN dataset returns (low_res, high_res)
        # We want to save the high_res as the image, with a dummy label
        low_res, high_res = self.sr_dataset[idx]
        return high_res, 0  # (image, label) format

def main(restart=False, best_fid=float('inf'), best_fid_epoch=0):
    start_epoch = 1
    wandb = createWandB()
    utils.prepare_data()
    loader = DataLoader(load_datapairs(), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.num_workers > 0)
    gen = Generator(in_channels=cfg.num_channels).to(cfg.device)
    disc = Discriminator(in_channels=cfg.num_channels).to(cfg.device)
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if restart:
        # Check for existing checkpoint to resume training
        start_epoch = utils.reload_checkpoint_model(gen, disc, opt_gen, opt_disc)

    if start_epoch == 1:
        print("Starting training from scratch")
    else:
        print(f"Resuming training from epoch {start_epoch}")

    opt_disc, opt_gen, best_fid, best_fid_epoch = training_loop(loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb, start_epoch, best_fid, best_fid_epoch)
    utils.save_model(gen, disc, opt_gen, opt_disc, f"epoch:{cfg.num_epochs}", "sr_gan_checkpoint.pth")

    real_images_dir = cfg.FID_REAL_DIR
    generated_images_dir = cfg.FID_FAKE_DIR
    validation_dataset = utils.build_fid_evaluation_dataset(load_fn=load_datapairs)
    real_count = len(validation_dataset)
    # Log dataset sizes to WandB
    wandb.log({
        "dataset/train_size": len(utils.load_data(split="train")),
        "dataset/validation_size": real_count,
        "dataset/epoch": 0  # Log at epoch 0 for final evaluation
    }, commit=True)
    utils.save_real_images_metrics(SRGANRealImageWrapper(validation_dataset), real_images_dir)
    # Need to reload just the generator for final evaluation - always load best model
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    best_model_path = f"{gan_checkpoints_dir}/best_gan.pth"
    utils.load_best_model(gen, best_model_path)
    utils.generate_images_metrics(gen, generated_images_dir, cfg.num_images_fid_score)
    fid_value = metrics.calc_fid_score(real_images_dir, generated_images_dir)
    kid_mean, kid_std = metrics.calc_kid_score(real_images_dir, generated_images_dir)
    print(f"FID score: {fid_value}")
    print(f"KID mean: {kid_mean:.6f}, KID std: {kid_std:.6f}")
    uploadLogsAndMetricsToWandB(wandb)
    # Log final metrics
    wandb.run.summary["final_fid"] = fid_value
    wandb.run.summary["final_kid_mean"] = kid_mean
    wandb.run.summary["final_kid_std"] = kid_std
    # Log validation metrics to CSV
    utils.log_validation_row(val_csv_path,
                            best_fid_epoch,
                            best_fid,  # This is the best FID observed during training
                            fid_value,  # This is the final FID from evaluation of best model
                            kid_mean,
                            kid_std)
    wandb.finish()

if __name__ == '__main__':
    main()