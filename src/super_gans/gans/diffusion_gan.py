import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid, save_image

import super_gans.config as cfg
import wandb
from super_gans import utils
from super_gans import metrics

# Enable cuDNN auto‑tuner to pick the fastest convolution algorithms for fixed‑size inputs
torch.backends.cudnn.benchmark = True
# Allow TensorFloat‑32 (TF32) on matrix multiplications and cuDNN operations (Ampere+ GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return betas_clipped


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return np.linspace(beta_start, beta_end, timesteps)


class DiffusionUNet(nn.Module):
    """
    A simple U-Net for noise prediction.
    Adapted for 128x128 grayscale images.
    """

    def __init__(self, image_size=128, in_channels=1, out_channels=1, time_emb_dim=100):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # Downsampling
        self.down1 = self._conv_block(64, 128)
        self.down2 = self._conv_block(128, 256)
        self.down3 = self._conv_block(256, 512)

        # Upsampling
        self.up1 = self._up_conv_block(512, 256)
        self.up2 = self._up_conv_block(256, 128)
        self.up3 = self._up_conv_block(128, 64)

        # Final convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def _up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t):
        """
        x: (batch, channels, height, width)
        t: (batch, 1) or (batch,) - time steps
        """
        # Time embedding
        t = t.float() / cfg.diffusion_timesteps  # Normalize to [0, 1]
        t = t.unsqueeze(-1)  # (batch, 1)
        time_emb = self.time_mlp(t)  # (batch, time_emb_dim)
        # We'll add time_emb to each feature map via adaptive average pooling and broadcasting
        # For simplicity, we'll use a method that adds time_emb to the input of each block
        # But note: our UNet doesn't have a straightforward way to add time_emb to mid-layers.
        # We'll use a simple approach: add time_emb to the initial conv output and then
        # let the network learn to use it. Alternatively, we can use FiLM or similar.
        # For now, we'll just concatenate or add after the initial conv.
        # We'll do: init_conv(x) + time_emb projected to channels and spatial dimensions.
        # However, to keep it simple and similar to common implementations, we'll
        # use the time embedding to modulate the features via AdaGN (not implemented here due to complexity).
        # Instead, we'll add the time embedding as a bias after the initial conv and
        # then let the network learn. This is a simplification.

        # We'll project time_emb to have the same number of channels as the initial conv output (64)
        # and then add it as a bias (after expanding to spatial dimensions).
        time_emb = self.time_mlp(t)  # (batch, time_emb_dim)
        time_emb = nn.Linear(self.time_emb_dim, 64)(time_emb)  # (batch, 64)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)  # (batch, 64, 1, 1)

        # Initial conv
        x = self.init_conv(x)  # (batch, 64, H, W)
        x = x + time_emb  # Add time embedding

        # Downsampling
        x1 = self.down1(x)  # (batch, 128, H/2, W/2)
        x2 = self.down2(x1)  # (batch, 256, H/4, W/4)
        x3 = self.down3(x2)  # (batch, 512, H/8, W/8)

        # Upsampling
        x = self.up1(x3)  # (batch, 256, H/4, W/4)
        x = x + x2  # Skip connection
        x = self.up2(x)  # (batch, 128, H/2, W/2)
        x = x + x1  # Skip connection
        x = self.up3(x)  # (batch, 64, H, W)
        x = x + self.init_conv(x)  # Skip connection from initial (with time embedding) - note: we already added time_emb to init_conv output

        # Final conv
        output = self.final_conv(x)  # (batch, 1, H, W)
        return output


def extract(a, t, x_shape):
    """
    Extract coefficients from a based on t and reshape to match x_shape.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Diffusion:
    """
    Diffusion model class to handle the forward and reverse processes.
    """

    def __init__(self, timesteps=cfg.diffusion_timesteps, beta_start=cfg.diffusion_beta_start, beta_end=cfg.diffusion_beta_end):
        self.timesteps = timesteps

        # Beta schedule
        self.betas = torch.from_numpy(linear_beta_schedule(timesteps, beta_start, beta_end)).float()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (compute x_t given x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x_t, t):
        """
        Sample x_{t-1} from x_t using the model.
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the noise
        model_output = model(x_t, t)
        mean = sqrt_recip_alphas_t * (x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            # Algorithm 2 line 4:
            return mean + torch.sqrt(posterior_variance_t) * noise

    def sample(self, model, image_size, batch_size=16, channels=1):
        """
        Generate samples from the model.
        """
        device = next(model.parameters()).device
        img = torch.randn(batch_size, channels, image_size, image_size, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t)
        return img


def training_loop(
    model,
    diffusion,
    dataset,
    wandb,
    start_epoch=1,
    best_fid=float("inf"),
    best_fid_epoch=0,
):
    # Create data loader
    loader_kwargs = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'pin_memory': cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4
        loader_kwargs['persistent_workers'] = True

    loader = DataLoader(dataset, **loader_kwargs)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Define master timeline metric
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        process = psutil.Process(os.getpid())
        print(
            f"Epoch: {epoch} | RAM GB: {process.memory_info().rss / 1024**3:.2f} \
              | GPU GB: {torch.cuda.memory_allocated() / 1024**3:.2f}"
        )

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (real_orig, _) in enumerate(loader):
            real = real_orig.to(cfg.device)
            batch_size = real.shape[0]

            # Sample a random time step for each image in the batch
            t = torch.randint(0, cfg.diffusion_timesteps, (batch_size,), device=cfg.device).long()

            # Sample noise to add to the images
            noise = torch.randn_like(real)

            # Get the noisy image (x_t)
            x_noisy = diffusion.q_sample(real, t, noise=noise)

            # Predict the noise
            noise_pred = model(x_noisy, t)

            # Loss is MSE between predicted and actual noise
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

        # --- LOG LOSSES EVERY EPOCH ---
        wandb.log(
            {
                "Loss/Diffusion": avg_loss,
                "epoch": epoch,
            },
            commit=False,
        )

        # --- VISUALS AT START OF EPOCH ---
        if epoch == start_epoch or epoch % 10 == 0:
            print(f"Epoch [{epoch}/{cfg.num_epochs}] Loss: {avg_loss:.4f}")
            # Generate some samples to log to WandB
            model.eval()
            with torch.inference_mode():
                # Generate 16 images
                sampled_imgs = diffusion.sample(model, cfg.image_size, batch_size=16, channels=cfg.num_channels)
                sampled_imgs = sampled_imgs.clamp(-1, 1)
                # Convert to RGB for logging (repeat grayscale to 3 channels)
                sampled_imgs_rgb = sampled_imgs.repeat(1, 3, 1, 1)
                # Create a grid
                img_grid = make_grid(sampled_imgs_rgb, nrow=4, normalize=True, value_range=(-1, 1))
                wandb.log({"Generated Samples": wandb.Image(img_grid, caption=f"epoch_{epoch:03d}"), "epoch": epoch})
            model.train()

        # --- FID CALCULATION AT END OF EPOCH ---
        if (epoch == start_epoch) or (epoch % cfg.fid_interval == 0) or (epoch == cfg.num_epochs):
            # Save current model as latest
            utils.save_model(model, model, optimizer, optimizer, epoch, filename="latest_diffusion.pth")
            # For FID, we need to generate a set of images and compare to real images
            # We'll generate cfg.num_images_fid_score images
            model.eval()
            with torch.inference_mode():
                # Generate images in batches to avoid OOM
                generated_imgs = []
                for i in range(0, cfg.num_images_fid_score, cfg.batch_size):
                    current_batch = min(cfg.batch_size, cfg.num_images_fid_score - i)
                    batch_imgs = diffusion.sample(model, cfg.image_size, batch_size=current_batch, channels=cfg.num_channels)
                    generated_imgs.append(batch_imgs)
                generated_imgs = torch.cat(generated_imgs, dim=0)
                generated_imgs = generated_imgs.clamp(-1, 1)

                # Save generated images to temporary directory for FID calculation
                generated_dir = f"{cfg.RESULTS_DIR}/generated_images_fid"
                os.makedirs(generated_dir, exist_ok=True)
                for img_idx, img in enumerate(generated_imgs):
                    # Convert grayscale to RGB by repeating channels
                    img_rgb = img.repeat(3, 1, 1)
                    save_image(img_rgb, f"{generated_dir}/generated_{img_idx:05d}.png", normalize=True, value_range=(-1, 1))

                # Use the validation dataset for real images (to avoid data leakage)
                val_dataset = utils.build_fid_evaluation_dataset()
                real_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
                os.makedirs(real_dir, exist_ok=True)
                # Save real images if not already saved (we can overwrite)
                utils.save_real_images_metrics(val_dataset, real_dir)

                # Calculate FID
                fid_value = metrics.calc_fid_score(real_dir, generated_dir)
                print(f"--- Epoch [{epoch}] FID Score: {fid_value:.4f} ---")

                # Log to WandB
                wandb.log({"Metrics/FID": fid_value, "epoch": epoch}, commit=True)

                # Checkpoint: Save as 'best' if FID improved (lower is better)
                if fid_value < best_fid:
                    best_fid = fid_value
                    best_fid_epoch = epoch
                    utils.save_model(model, model, optimizer, optimizer, epoch, filename="best_diffusion.pth")
                    print(f"*** New best FID: {best_fid:.4f} at epoch {best_fid_epoch} ***")
            model.train()
        else:
            # Commit the losses and move the custom timeline forward on non-FID epochs
            wandb.log({"epoch": epoch}, commit=True)

        # End of Epoch cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # --- FINAL RUN SUMMARY ---
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid

    return optimizer, best_fid, best_fid_epoch


def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("diffusion-model", type="model")
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    file_path = f"{gan_checkpoints_dir}/best_diffusion.pth"
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def createWandB():
    wandb.init(
        project="super-gans-project",
        name="Diffusion_GAN",
        config={
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "image_size": cfg.image_size,
            "num_channels": cfg.num_channels,
            "diffusion_timesteps": cfg.diffusion_timesteps,
            "diffusion_beta_start": cfg.diffusion_beta_start,
            "diffusion_beta_end": cfg.diffusion_beta_end,
        },
    )
    return wandb


def main(restart=False, best_fid=float("inf"), best_fid_epoch=0):
    # writer = SummaryWriter("logs/diffusion_run_1")
    wandb = createWandB()
    utils.prepare_data()
    train_dataset = utils.load_data()

    # Create the model
    model = DiffusionUNet(
        image_size=cfg.image_size,
        in_channels=cfg.num_channels,
        out_channels=cfg.num_channels
    ).to(cfg.device)
    # Compile model for faster training (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # Create the diffusion process
    diffusion = Diffusion(
        timesteps=cfg.diffusion_timesteps,
        beta_start=cfg.diffusion_beta_start,
        beta_end=cfg.diffusion_beta_end
    )

    start_epoch = 1
    if restart:
        # Check for existing checkpoint to resume training
        start_epoch = utils.reload_checkpoint_model(model, None, None, None)

    if start_epoch == 1:
        print("Starting training from scratch")
    else:
        print(f"Resuming training from epoch {start_epoch}")

    optimizer, best_fid, best_fid_epoch = training_loop(
        model, diffusion, train_dataset, wandb, start_epoch, best_fid, best_fid_epoch
    )
    # Save final checkpoint
    utils.save_model(
        model, model, optimizer, optimizer, f"epoch:{cfg.num_epochs}", "diffusion_checkpoint.pth"
    )

    real_images_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
    generated_images_dir = f"{cfg.RESULTS_DIR}/generated_images_fid"
    validation_dataset = utils.build_fid_evaluation_dataset()
    real_count = len(validation_dataset)
    # Log dataset sizes to WandB
    wandb.log({
        "dataset/train_size": len(utils.load_data(split="train")),
        "dataset/validation_size": real_count,
        "dataset/epoch": 0  # Log at epoch 0 for final evaluation
    }, commit=True)
    # Regenerate images for final evaluation (using best model)
    # Load best model
    best_model_path = f"{cfg.MODELS_DIR}/gan_checkpoints/best_diffusion.pth"
    utils.load_best_model(model, best_model_path)
    model.eval()
    with torch.inference_mode():
        # Generate images for final FID
        generated_imgs = []
        for i in range(0, cfg.num_images_fid_score, cfg.batch_size):
            current_batch = min(cfg.batch_size, cfg.num_images_fid_score - i)
            batch_imgs = diffusion.sample(model, cfg.image_size, batch_size=current_batch, channels=cfg.num_channels)
            generated_imgs.append(batch_imgs)
        generated_imgs = torch.cat(generated_imgs, dim=0)
        generated_imgs = generated_imgs.clamp(-1, 1)

        # Save generated images
        os.makedirs(generated_images_dir, exist_ok=True)
        for img_idx, img in enumerate(generated_imgs):
            img_rgb = img.repeat(3, 1, 1)
            save_image(img_rgb, f"{generated_images_dir}/generated_{img_idx:05d}.png", normalize=True, value_range=(-1, 1))

        # Save real images (if not already saved)
        os.makedirs(real_images_dir, exist_ok=True)
        utils.save_real_images_metrics(validation_dataset, real_images_dir)

        # Calculate final FID and KID
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
    wandb.finish()


if __name__ == "__main__":
    main()