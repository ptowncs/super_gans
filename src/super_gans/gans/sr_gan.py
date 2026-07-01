import psutil, os, gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import RandomSampler
from PIL import Image
import super_gans.config as cfg
from super_gans import utils
from torchvision.utils import *
from pytorch_fid import fid_score
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb
from torchvision.models import vgg19
from tqdm import tqdm

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

def train_fn(epoch, loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb):
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
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{cfg.num_epochs}] Loss D: {loss_disc.item():.4f}, Loss G: {gen_loss.item():.4f}")
            log_tensorboard_visuals(wandb, gen, high_res, low_res, epoch)


def training_loop(loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb, start_epoch=0, best_fid=float('inf'), best_fid_epoch=0):
    # feature=64 uses a lower layer of Inception; it's faster for monitoring
    fid_metric = FrechetInceptionDistance(feature=cfg.fid_dims, normalize=True).to(cfg.device)
    
    # Define master timeline metric
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")
    
    for epoch in range(cfg.num_epochs):
        process = psutil.Process(os.getpid())
        print(f"Epoch: {epoch} | RAM GB: {process.memory_info().rss / 1024**3:.2f} \
              | GPU GB: {torch.cuda.memory_allocated() / 1024**3:.2f}")
        train_fn(epoch, loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb)
        
        
        # --- FID CALCULATION AT END OF EPOCH ---
        if (epoch % cfg.fid_interval == 0) or (epoch == cfg.num_epochs - 1):
            save_model(gen, disc, opt_gen, opt_disc, epoch, filename="latest_gan.pth")
            current_fid = calculate_fid_sample(gen, loader, fid_metric)
            fid_metric.reset()
            
            wandb.log({"Metrics/FID": current_fid, "epoch": epoch}, commit=True)
            print(f"--- Epoch [{epoch}] FID Score: {current_fid:.4f} ---")
        
            # Checkpoint: Save as 'best' if quality improved
            if current_fid < best_fid:
                best_fid = current_fid
                best_fid_epoch = epoch
                save_model(gen, disc, opt_gen, opt_disc, epoch, filename="best_gan.pth")
        else:
            wandb.log({"epoch": epoch}, commit=True)
        
        # End of Epoch cleanup
        #writer.flush()
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- FINAL RUN SUMMARY ---
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid

def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("dc-gan-model", type="model")
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    file_path = f"{gan_checkpoints_dir}/best_gan.pth"
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)

def save_model(gen, disc, opt_gen, opt_disc, epoch, filename="checkpoint.pth"):
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    os.makedirs(gan_checkpoints_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_G_state_dict": opt_gen.state_dict(),
        "optimizer_D_state_dict": opt_disc.state_dict(),
    }
    
    save_path = f"{gan_checkpoints_dir}/{filename}"
    torch.save(checkpoint, save_path)
    print(f"--- Saved checkpoint: {filename} at epoch {epoch} ---")

def reloadModel(filename="best_gan.pth"): # Default to the best one
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    generator = Generator().to(cfg.device)
    checkpoint_path = f"{gan_checkpoints_dir}/{filename}"

    try:
        # Map location ensures it loads on the right device (CPU/GPU)
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        # Use the correct key from your save_model function
        generator.load_state_dict(checkpoint["generator_state_dict"])
        print(f"Loaded {filename} weights.")
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        
    generator.eval()
    return generator

def calculate_fid_sample(gen, loader, fid_metric):
    """
    Standalone FID calculation for SR-GAN.
    Compares high-res ground truth images against generated super-res images.
    """
    gen.eval()
    fid_metric.reset()
    
    assert cfg.num_images_fid_sample % cfg.batch_size == 0, "FID sample count must be divisible by batch size"
    batch_size = cfg.batch_size
    n_batches = cfg.num_images_fid_sample // batch_size
    data_iter = iter(loader)

    fid_metric = fid_metric.to(cfg.device)
    
    with torch.inference_mode():
        for _ in range(n_batches):
            try:
                low_res_batch, high_res_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                low_res_batch, high_res_batch = next(data_iter)
                
            low_res_batch = low_res_batch[:batch_size].to(cfg.device)
            high_res_batch = high_res_batch[:batch_size].to(cfg.device)
           
            # 1. Process High-Res (Real) Images
            real_rgb = (high_res_batch.expand(batch_size, 3, -1, -1) + 1.0) / 2.0
            fid_metric.update(real_rgb, real=True)

            # 2. Process Super-Res (Fake) Images mapping low-res straight to generator
            fake_batch = gen(low_res_batch)
            fake_rgb = (fake_batch.expand(batch_size, 3, -1, -1) + 1.0) / 2.0
            fid_metric.update(fake_rgb, real=False)

    fid_score_val = fid_metric.compute().item()

    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    gen.train()
    return fid_score_val

def log_tensorboard_visuals(wandb, gen, real_batch, gen_input, epoch):
    """
    Captures the current state of generation vs real images.
    """
    gen.eval() 
    with torch.inference_mode():
        # 1. Generate fakes (Shape: N, 1, H, W)
        fake = gen(gen_input).reshape(-1, cfg.num_channels, cfg.image_size, cfg.image_size)
        
        # 2. Reshape real data (Shape: N, 1, H, W)
        real = real_batch.reshape(-1, cfg.num_channels, cfg.image_size, cfg.image_size)

        # 3. Convert both from 1-channel to 3-channel (RGB)
        # This is necessary so the grid looks consistent in all viewers

        #fake_rgb = fake.repeat(1, 3, 1, 1)
        #real_rgb = real.repeat(1, 3, 1, 1)

        fake_rgb = fake.expand(-1, 3, -1, -1)
        real_rgb = real.expand(-1, 3, -1, -1)

        # 4. Create grids using Torchvision's built-in normalization
        # normalize=True: shifts the range to [0, 1]
        # value_range=(-1, 1): tells the function our Tanh/Transform output is [-1, 1]
        img_grid_fake = torchvision.utils.make_grid(fake_rgb, nrow=8, normalize=True, value_range=(-1, 1))
        img_grid_real = torchvision.utils.make_grid(real_rgb, nrow=8, normalize=True, value_range=(-1, 1))
        # 5. Log to TensorBoard
        #writer.add_image("Images/Generated", img_grid_fake, global_step=epoch)
        #writer.add_image("Images/Real", img_grid_real, global_step=epoch)
        # --- Make grids ---
        
        # --- Log to WandB ---
        wandb.log({"Generated Grid": wandb.Image(img_grid_fake, caption=f"epoch_{epoch:03d}"),
                   "Real Grid": wandb.Image(img_grid_real, caption=f"epoch_{epoch:03d}"),
                   "epoch": epoch}, commit=False)
    gen.train()

def createWandB():
    wandb.init(
        project="super-gans-project",
        name="dcgan_run_1",
        config={
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "z_dim": cfg.z_dim,
            "image_size": cfg.image_size,
            "num_channels": cfg.num_channels,
            })
    return wandb

def load_datapairs(split="train"):
    class DataSetWithHiResLowResPair(Dataset):
        def __init__(self, root_dir):
            super().__init__()
            self.data = []
            self.root_dir = root_dir
            self.class_names = os.listdir(root_dir)

            for index, name in enumerate(self.class_names):
                files = os.listdir(os.path.join(root_dir, name))
                self.data += list(zip(files, [index] * len(files)))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            img_file, label = self.data[index]
            root_and_dir = os.path.join(self.root_dir, self.class_names[label])
            # Fix: Force image to 1-channel Grayscale ("L" mode) right after opening
            image = np.array(Image.open(os.path.join(root_and_dir, img_file)).convert("L"))
            image = cfg.both_transforms(image=image)["image"]
            high_res = cfg.high_res_transform(image=image)["image"]
            low_res = cfg.low_res_transform(image=image)["image"]
            return low_res, high_res
    
    dataset_path = utils.get_dataset_path("paultimothymooney/chest-xray-pneumonia/versions/2")
    chest_xray_ds =  f"{dataset_path}/chest_xray"
    dataset = DataSetWithHiResLowResPair(root_dir=f"{chest_xray_ds}/{split}")
    return dataset

def save_images_fid(dataset, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    for i in range(min(cfg.num_images_fid_score, len(dataset))):
        # Unpack the paired dataset: we want the high_res ground truth image
        low_res, high_res = dataset[i] 

        # Convert grayscale -> RGB by repeating channels
        image_rgb = high_res.repeat(3, 1, 1)
        filename = os.path.join(to_dir, f"pneumonia_{i:04d}.png")
        
        # Saves the ground truth target scaled correctly
        save_image(image_rgb, filename, normalize=True, value_range=(-1, 1))

    print(
        f"Saved {min(cfg.num_images_fid_score, len(dataset))} real high-res Pneumonia images to {to_dir}/"
    )

def generate_images_fid(generator, dataset, generated_images_dir, batch_size=128):
    os.makedirs(generated_images_dir, exist_ok=True)
    generator.eval()
    
    # Use a simple dataloader to stream low-res inputs in chunks
    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    
    images_saved = 0
    
    with torch.inference_mode():
        for low_res_batch, _ in eval_loader:
            # Safety check to stop exactly at your requested evaluation count
            if images_saved >= cfg.num_images_fid_score:
                break
                
            low_res_batch = low_res_batch.to(cfg.device)
            
            # Generate super-resolution images using the low-res inputs
            generated_images = generator(low_res_batch)
            
            for img in generated_images:
                if images_saved >= cfg.num_images_fid_score:
                    break
                    
                # Convert Grayscale -> RGB to match the real images directory
                image_rgb = img.repeat(3, 1, 1)
                filename = os.path.join(generated_images_dir, f"generated_image_{images_saved:04d}.png")
                save_image(image_rgb, filename, normalize=True, value_range=(-1, 1))
                images_saved += 1

    print(f"Successfully generated and saved {images_saved} images to {generated_images_dir}/")

def main():
    wandb = createWandB()
    loader = DataLoader(load_datapairs(), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    gen = Generator(in_channels=cfg.num_channels).to(cfg.device)
    disc = Discriminator(in_channels=cfg.num_channels).to(cfg.device)
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    training_loop(loader, disc, gen, opt_disc, opt_gen, mse, bce, vgg_loss, wandb)
    save_model(gen, disc, opt_gen, opt_disc, f"epoch:{cfg.num_epochs}", "sr_gan_checkpoint.pth")
    
    real_images_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
    generated_images_dir = f"{cfg.RESULTS_DIR}/fake_images_fid"
    fid_dataset = utils.build_fid_evaluation_dataset(load_fn=load_datapairs, target_samples=cfg.num_images_fid_score)
    save_images_fid(fid_dataset, real_images_dir)
    best_model = reloadModel("best_gan.pth")
    generate_images_fid(best_model, fid_dataset, generated_images_dir, batch_size=cfg.batch_size)
    fid_value = utils.calc_fid_score(real_images_dir, generated_images_dir)
    print(f"FID score: {fid_value}")
    uploadLogsAndMetricsToWandB(wandb)
    # Log final metrics
    wandb.run.summary["final_fid"] = fid_value
    wandb.finish()

if __name__ == '__main__':
    main()