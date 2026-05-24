import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import RandomSampler
from PIL import Image  # Import PIL Image
import super_gans.config as cfg
from super_gans import utils
from torchvision.utils import save_image
from pytorch_fid import fid_score
from torchmetrics.image.fid import FrechetInceptionDistance
import wandb
import gc
import psutil, os, torch

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = cfg.image_dim
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),  # Analyzes image features
            nn.LeakyReLU(0.01),  # Prevents dead neurons
            nn.Linear(128, 1),  # Outputs a single number
            nn.Sigmoid(),  # Converts into probability score
        )

    def forward(self, x):
        return self.disc(x)  # Defines how data flows through the network


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # z_dim is size of noise vector , image_dim is total pixels in image
        self.gen = nn.Sequential(
            nn.Linear(cfg.z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, cfg.image_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


def training_loop(disc, gen, dataset, wandb):
    fixed_noise = torch.randn((cfg.batch_size, cfg.z_dim)).to(cfg.device)

    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr)
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()
    
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    # 'step' tracks total batches seen (X-axis for loss charts)
    step = 0
    # feature=64 uses a lower layer of Inception; it's faster for monitoring
    # Moving to CPU to avoid GPU contention with GANs
    fid_metric = FrechetInceptionDistance(feature=cfg.fid_dims, normalize=True).to(cfg.device)
    best_fid = float('inf') # Initialize with infinity
    best_fid_epoch = 0
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")
    for epoch in range(cfg.num_epochs):
        process = psutil.Process(os.getpid())
        print(f"Epoch: {epoch} | RAM GB: {process.memory_info().rss / 1024**3:.2f} \
              | GPU GB: {torch.cuda.memory_allocated() / 1024**3:.2f}")

        for batch_idx, (real_orig, _) in enumerate(loader):
            real = real_orig.view(-1, cfg.image_dim).to(cfg.device)
            batch_size = real.shape[0]

            ### Train Discriminator ###
            disc.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, cfg.z_dim).to(cfg.device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake.detach(), torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            # Backpropagation
            lossD.backward()
            opt_disc.step()

            ### Train Generator ###
            gen.zero_grad(set_to_none=True)
            lossG = criterion(disc_fake, torch.ones_like(output))
            lossG.backward()
            opt_gen.step()
            step += 1 # Increment every batch for smooth loss curves
        
        # --- LOG LOSSES EVERY EPOCH ---
        #writer.add_scalar("Loss/Discriminator", lossD.item(), global_step=epoch)
        #writer.add_scalar("Loss/Generator", lossG.item(), global_step=epoch)
        wandb.log({"Loss/Discriminator": lossD.item(), "Loss/Generator": lossG.item(), "epoch": epoch}, commit=False)
            

        # --- VISUALS AT START OF EPOCH ---
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{cfg.num_epochs}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")
            log_tensorboard_visuals(wandb, gen, real_orig, fixed_noise, epoch)

        # --- FID CALCULATION AT END OF EPOCH ---
        if (epoch % cfg.fid_interval == 0) or (epoch == cfg.num_epochs - 1):
            save_model(gen, disc, opt_gen, opt_disc, epoch, filename="latest_gan.pth")
            current_fid = utils.calculate_fid_sample(gen, loader, fid_metric)
            fid_metric.reset()
            #writer.add_scalar("Metrics/FID", current_fid, global_step=epoch)
            wandb.log({"Metrics/FID": current_fid}, step=epoch)
            print(f"--- Epoch [{epoch}] FID Score: {current_fid:.4f} ---")
        
            # Checkpoint: Save as 'best' if quality improved
            if current_fid < best_fid:
                best_fid = current_fid
                best_fid_epoch = epoch
                save_model(gen, disc, opt_gen, opt_disc, epoch, filename="best_gan.pth")
        else:
            wandb.log({}, commit=True)
        # End of Epoch cleanup
        #writer.flush()
        torch.cuda.empty_cache()
        gc.collect()
        
    
    #writer.add_text('Final Results', f'Best FID Epoch: {best_fid_epoch}, Best FID Sample Score: {best_fid}')
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid
    #writer.flush()
    
    
    return opt_disc, opt_gen

def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("simple-gan-model", type="model")
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

def log_tensorboard_visuals(wandb, gen, real_batch, fixed_noise, epoch):
    """
    Captures the current state of generation vs real images.
    """
    gen.eval() 
    with torch.inference_mode():
        # 1. Generate fakes (Shape: N, 1, H, W)
        fake = gen(fixed_noise).reshape(-1, cfg.num_channels, cfg.image_size, cfg.image_size)
        
        # 2. Reshape real data (Shape: N, 1, H, W)
        real = real_batch.reshape(-1, cfg.num_channels, cfg.image_size, cfg.image_size)

        # 3. Convert both from 1-channel to 3-channel (RGB)
        # This is necessary so the grid looks consistent in all viewers
        fake_rgb = fake.repeat(1, 3, 1, 1)
        real_rgb = real.repeat(1, 3, 1, 1)

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
        name="gan_run_1",
        config={
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "z_dim": cfg.z_dim,
            "image_size": cfg.image_size,
            "num_channels": cfg.num_channels,
            })
    return wandb

if __name__ == '__main__':
    #writer = SummaryWriter("logs/simple_gan_run_1")
    wandb = createWandB()
    train_dataset = utils.load_data()
    disc = Discriminator().to(cfg.device)
    gen = Generator().to(cfg.device)
    opt_disc, opt_gen = training_loop(disc, gen, train_dataset,wandb)
    save_model(gen, disc, opt_gen, opt_disc, f"epoch:{cfg.num_epochs}", "simple_gan_checkpoint.pth")
    
    real_images_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
    generated_images_dir = f"{cfg.RESULTS_DIR}/fake_images_fid"
    fid_dataset = utils.build_fid_evaluation_dataset(cfg.num_images_fid_score)
    utils.save_images_fid(fid_dataset, real_images_dir)
    best_model = reloadModel("best_gan.pth")
    #best_model = reloadModel("best_gan.pth")
    utils.generate_images_fid(best_model, generated_images_dir)
    fid_value = utils.calc_fid_score(real_images_dir, generated_images_dir)
    print(f"FID score: {fid_value}")
    # writer.close()
    uploadLogsAndMetricsToWandB(wandb)
    # Log final metrics
    wandb.run.summary["final_fid"] = fid_value
    wandb.finish()

