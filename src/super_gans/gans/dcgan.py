import gc
import os

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image  # Import PIL Image
from pytorch_fid import fid_score

# from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import *

import super_gans.config as cfg
import wandb
from super_gans import utils
from super_gans import metrics

# Enable cuDNN auto‑tuner to pick the fastest convolution algorithms for fixed‑size inputs
import torch
torch.backends.cudnn.benchmark = True
# Allow TensorFloat‑32 (TF32) on matrix multiplications and cuDNN operations (Ampere+ GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def initialize_weights(model):
    # DCGAN‑style weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class Discriminator(nn.Module):
    """
    Discriminator: increases the number of channels(features) while downsampling the spatial dimensions.
    Architecture follows the DCGAN paper (adjusted for 128x128 input).
    """

    def __init__(self, features_d=64):
        super(Discriminator, self).__init__()
        self.features_d = features_d

        def _block(in_channels, out_channels, kernel_size, stride, padding):
            # Note: The reference implementation omits BatchNorm and uses non‑inplace LeakyReLU.
            # We keep bias=False as in the reference.
            # INTENTIONAL DEVIATION: We omit BatchNorm2d layers throughout the discriminator
            # to match the reference implementation from Aladdin Persson's Machine Learning Collection.
            # This deviates from some DCGAN variants but provides more stable training for our specific use case.
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                ),
                # nn.BatchNorm2d(out_channels),  # intentionally omitted to match reference
                nn.LeakyReLU(0.2),
            )

        self.disc = nn.Sequential(
            # input: N x cfg.num_channels x 128 x 128
            nn.Conv2d(cfg.num_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            _block(features_d, features_d * 2, 4, 2, 1),   # -> 32x32
            _block(features_d * 2, features_d * 4, 4, 2, 1), # -> 16x16
            _block(features_d * 4, features_d * 8, 4, 2, 1), # -> 8x8
            _block(features_d * 8, features_d * 16, 4, 2, 1),# -> 4x4
            # After the above blocks we have 4x4 feature map
            nn.Conv2d(features_d * 16, 1, 4, 1, 0, bias=False), # -> 1x1
            # No Sigmoid because we use BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """
    Generator: increases spatial dimensions while decreasing channel depth.
    Architecture follows the DCGAN paper (adjusted for 128x128 output).
    An explicit upsample layer is kept as a safety‑net for experimenting with
    different image sizes; this deviates from the strict reference but adds
    virtually no cost and protects against off‑by‑one errors.
    """

    def __init__(self, z_dim=128, features_g=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.features_g = features_g

        def _block(in_channels, out_channels, kernel_size, stride, padding):
            # Note: The reference implementation omits BatchNorm and uses non‑inplace ReLU.
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                ),
                # nn.BatchNorm2d(out_channels),  # intentionally omitted to match reference
                nn.ReLU(),
            )

        # Build: z_dim -> features_g*16 x 4x4
        self.net = nn.Sequential(
            _block(z_dim, features_g * 16, 4, 1, 0),   # 4x4
            _block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            _block(features_g * 8, features_g * 4, 4, 2, 1),   # 16x16
            _block(features_g * 4, features_g * 2, 4, 2, 1),   # 32x32
            _block(features_g * 2, features_g, 4, 2, 1),       # 64x64
            nn.ConvTranspose2d(features_g, cfg.num_channels, 4, 2, 1, bias=False),
        )
        # Safety upsample to guarantee exact output size when experimenting
        self.upsample = nn.Upsample(size=(cfg.image_size, cfg.image_size), mode='nearest')
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.net(x)
        x = self.upsample(x)
        return self.tanh(x)


def training_loop(
    disc,
    gen,
    opt_disc,
    opt_gen,
    dataset,
    wandb,
    start_epoch=1,
    best_fid=float("inf"),
    best_fid_epoch=0,
):
    fixed_noise = torch.randn((cfg.batch_size, cfg.z_dim)).to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()

    loader_kwargs = {
        'dataset': dataset,
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'pin_memory': cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4
        loader_kwargs['persistent_workers'] = True

    loader = DataLoader(**loader_kwargs)
    # feature=64 uses a lower layer of Inception; it's faster for monitoring
    fid_metric = FrechetInceptionDistance(feature=cfg.fid_dims, normalize=True).to(
        cfg.device
    )

    # Define master timeline metric
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")

    # Setup CSV logging
    train_csv_path = utils.setup_training_csv("dcgan")
    val_csv_path = utils.setup_training_csv("dcgan")  # Using same CSV for simplicity

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        process = psutil.Process(os.getpid())
        print(
            f"Epoch: {epoch} | RAM GB: {process.memory_info().rss / 1024**3:.2f} \
              | GPU GB: {torch.cuda.memory_allocated() / 1024**3:.2f}"
        )

        for batch_idx, (real_orig, _) in enumerate(loader):
            real = real_orig.to(cfg.device)
            batch_size = real.shape[0]

            ### Train Discriminator ###
            disc.zero_grad(set_to_none=True)

            # Pass 1: Discriminator updates on Real Images
            disc_real = disc(real).view(-1)

            # Apply One-Sided Label Smoothing to keep gradients healthy
            lossD_real = criterion(disc_real, torch.ones_like(disc_real) * 0.9)

            # Pass 2: Discriminator updates on Fake Images
            noise = torch.randn(batch_size, cfg.z_dim).to(cfg.device)
            fake = gen(noise)
            # Explicitly detach the fake images so backpropagation doesn't leak into the Generator
            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2
            lossD.backward()
            opt_disc.step()

            ### Train Generator ###
            opt_gen.zero_grad(set_to_none=True)

            # Pass 3: Evaluate fake images again with the updated Discriminator weights
            disc_fake_for_gen = disc(fake).view(-1)  # Do NOT detach here
            lossG = criterion(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))

            lossG.backward()
            opt_gen.step()

            # --- VISUALS AT START OF EPOCH ---
            if batch_idx == 0 and (epoch == start_epoch or epoch % 10 == 0):
                print(
                    f"Epoch [{epoch}/{cfg.num_epochs}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}"
                )
                utils.log_tensorboard_visuals(wandb, gen, real, fixed_noise, epoch)

        # --- LOG LOSSES EVERY EPOCH ---
        # Staging loss data. commit=False ensures we wait to push until the end of the epoch.
        wandb.log(
            {
                "Loss/Discriminator": lossD.item(),
                "Loss/Generator": lossG.item(),
                "epoch": epoch,
            },
            commit=False,
        )

        # --- FID CALCULATION AT END OF EPOCH ---
        if (epoch == start_epoch) or (epoch % cfg.fid_interval == 0) or (epoch == cfg.num_epochs):
            utils.save_model(
                gen, disc, opt_gen, opt_disc, epoch, filename="latest_gan.pth"
            )
            current_fid = metrics.compute_fid_from_images(gen, loader, fid_metric)
            fid_metric.reset()

            # VERIFIED FIX: Bundle 'epoch' into the dictionary and commit the full row to WandB
            wandb.log({"Metrics/FID": current_fid, "epoch": epoch}, commit=True)
            print(f"--- Epoch [{epoch}] FID Score: {current_fid:.4f} ---")

            # Log losses and FID to training CSV
            utils.log_training_row(train_csv_path, epoch,
                                  loss_g=lossG.item(),
                                  loss_d=lossD.item(),
                                  fid_train=current_fid)
            print(f"Epoch [{epoch}/{cfg.num_epochs}] CSV: G_loss={lossG.item():.6f}, D_loss={lossD.item():.6f}, FID_train={current_fid:.6f}")

            # Checkpoint: Save as 'best' if quality improved
            if current_fid < best_fid:
                best_fid = current_fid
                best_fid_epoch = epoch
                utils.save_model(
                    gen, disc, opt_gen, opt_disc, epoch, filename="best_gan.pth"
                )
            print(f"Epoch [{epoch}/{cfg.num_epochs}]: Best_fid_score={best_fid:.4f}, Best_fid_epoch={best_fid_epoch}")
        else:
            # VERIFIED FIX: Commits the losses and moves the custom timeline forward on non-FID epochs
            wandb.log({"epoch": epoch}, commit=True)
            # Log losses to training CSV (no FID)
            utils.log_training_row(train_csv_path, epoch,
                                  loss_g=lossG.item(),
                                  loss_d=lossD.item(),
                                  fid_train=None)
            print(f"Epoch [{epoch}/{cfg.num_epochs}] CSV: G_loss={lossG.item():.6f}, D_loss={lossD.item():.6f}, FID_train=N/A")

        # End of Epoch cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # --- FINAL RUN SUMMARY ---
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid

    return opt_disc, opt_gen, best_fid, best_fid_epoch


def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("dc-gan-model", type="model")
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    file_path = f"{gan_checkpoints_dir}/best_gan.pth"
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def createWandB():
    wandb.init(
        project="super-gans-project",
        name="DC_GAN",
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


def main(restart=False, best_fid=float("inf"), best_fid_epoch=0):
    # writer = SummaryWriter("logs/dcgan_run_1")
    wandb = createWandB()
    utils.prepare_data()
    train_dataset = utils.load_data()

    disc = Discriminator().to(cfg.device)
    gen = Generator().to(cfg.device)
    # Apply DCGAN weight initialization
    disc.apply(initialize_weights)
    gen.apply(initialize_weights)
    # Compile models for faster training (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        disc = torch.compile(disc)
        gen = torch.compile(gen)
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr, betas=cfg.betas)

    start_epoch = 1
    if restart:
        # Check for existing checkpoint to resume training
        start_epoch = utils.reload_checkpoint_model(gen, disc, opt_gen, opt_disc)

    if start_epoch == 1:
        print("Starting training from scratch")
    else:
        print(f"Resuming training from epoch {start_epoch}")

    opt_disc, opt_gen, best_fid, best_fid_epoch = training_loop(
        disc, gen, opt_disc, opt_gen, train_dataset, wandb, start_epoch, best_fid, best_fid_epoch
    )
    utils.save_model(
        gen, disc, opt_gen, opt_disc, f"epoch:{cfg.num_epochs}", "dc_gan_checkpoint.pth"
    )

    real_images_dir = cfg.FID_REAL_DIR
    generated_images_dir = cfg.FID_FAKE_DIR
    validation_dataset = utils.build_fid_evaluation_dataset()
    real_count = len(validation_dataset)
    # Log dataset sizes to WandB
    wandb.log({
        "dataset/train_size": len(utils.load_data(split="train")),
        "dataset/validation_size": real_count,
        "dataset/epoch": 0  # Log at epoch 0 for final evaluation
    }, commit=True)
    utils.save_real_images_metrics(validation_dataset, real_images_dir)
    # Need to reload just the generator for final evaluation - always load best model
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    best_model_path = f"{gan_checkpoints_dir}/best_gan.pth"
    utils.load_best_model(gen, best_model_path)
    utils.generate_images_metrics(gen, generated_images_dir, cfg.num_images_fid_score)
    fid_value = metrics.calc_fid_score(real_images_dir, generated_images_dir)
    kid_mean, kid_std = metrics.calc_kid_score(real_images_dir, generated_images_dir)
    print(f"FID score: {fid_value}")
    print(f"KID mean: {kid_mean:.6f}, KID std: {kid_std:.6f}")
    # writer.close()
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


if __name__ == "__main__":
    main()