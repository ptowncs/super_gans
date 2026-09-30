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


class Critic(nn.Module):
    """
    WGAN Critic (Discriminator): Instead of outputting a probability,
    outputs a scalar score. No sigmoid activation at the end.
    Architecture follows the DCGAN critic pattern but adapted for configurable image size (power of 2).
    """

    def __init__(self, channels=64, image_size=128):
        super(Critic, self).__init__()
        self.channels = channels
        self.image_size = image_size

        def _block(in_channels, out_channels, kernel_size, stride, padding, norm=True):
            # Following DCGAN paper: BatchNorm and bias=False for conv layers
            # Note: First layer of Critic does not use BatchNorm
            # Note: Using BatchNorm2d for stability; some references (e.g., Aladdin) use InstanceNorm2d
            # Note: Final layer uses padding=0 to ensure proper 2x2->1x1 downsampling
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                )
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return nn.Sequential(*layers)

        # Validate image_size is power of 2
        assert image_size & (image_size - 1) == 0, "image_size must be power of 2"
        assert image_size >= 32, "image_size too small for architecture"

        # Calculate number of stride-2 layers needed: log2(image_size)
        import math
        num_stride_2_layers = int(math.log2(image_size))

        layers = []

        # Initial layer: cfg.num_channels -> channels
        layers.append(nn.Conv2d(cfg.num_channels, channels, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2))

        # Intermediate _block layers: double channels each time
        curr_channels = channels
        # We need num_stride_2_layers - 2 intermediate _block layers
        # (1 initial + (num_stride_2_layers-2) intermediate + 1 final = num_stride_2_layers total)
        for i in range(num_stride_2_layers - 2):
            out_channels = curr_channels * 2
            layers.append(_block(curr_channels, out_channels, 4, 2, 1, norm=True))
            curr_channels = out_channels

        # Final layer to get to 1x1 output
        layers.append(nn.Conv2d(curr_channels, 1, 2, 2, 0, bias=False))
        # No Sigmoid because we use the Wasserstein loss

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):
    """
    WGAN Generator: Same as DCGAN generator but without the final Sigmoid
    (though we keep Tanh for consistency with [-1,1] range).
    Architecture follows the DCGAN generator pattern but adapted for configurable image size (power of 2).
    The architecture is built dynamically so no safety upsample is needed for power-of-2 sizes.
    """

    def __init__(self, z_dim=128, channels=64, image_size=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.channels = channels
        self.image_size = image_size

        def _block(in_channels, out_channels, kernel_size, stride, padding, norm=True):
            # Following DCGAN paper: BatchNorm and bias=False for conv layers
            # Note: Last layer of Generator does not use BatchNorm
            # Note: Using BatchNorm2d for stability; some references (e.g., Aladdin) use InstanceNorm2d
            layers = [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                )
            ]
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=False))
            return nn.Sequential(*layers)

        # Validate image_size is power of 2
        assert image_size & (image_size - 1) == 0, "image_size must be power of 2"
        assert image_size >= 32, "image_size too small for architecture"

        # Calculate number of stride-2 layers needed: log2(image_size) - 2
        import math
        num_stride_2_layers = int(math.log2(image_size)) - 2

        layers = []

        # Start with z_dim -> channels*16 x 4 x 4
        layers.append(_block(z_dim, channels * 16, 4, 1, 0, norm=True))  # 4x4

        # Intermediate _block layers: double spatial size, halve channels
        curr_channels = channels * 16
        # We need num_stride_2_layers - 1 intermediate _block layers
        # (1 initial + (num_stride_2_layers-1) intermediate + 1 final = num_stride_2_layers total)
        for i in range(num_stride_2_layers - 1):
            out_channels = curr_channels // 2
            layers.append(_block(curr_channels, out_channels, 4, 2, 1, norm=True))
            curr_channels = out_channels

        # Final layer to get to target image channels
        layers.append(nn.ConvTranspose2d(curr_channels, cfg.num_channels, 4, 2, 1, bias=False))

        # Optional: upsample to exact size if needed (shouldn't be needed for powers of 2, but keeping for safety)
        if 4 * (2 ** num_stride_2_layers) != image_size:
            self.upsample = nn.Upsample(size=(image_size, image_size), mode='nearest')
        else:
            self.upsample = None

        self.main = nn.Sequential(*layers)
        self.tanh = nn.Tanh()

    def forward(self, input):
        # Reshape input from [batch_size, z_dim] to [batch_size, z_dim, 1, 1] for ConvTranspose2d
        x = input.view(input.size(0), input.size(1), 1, 1)
        x = self.main(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return self.tanh(x)


def training_loop(
    critic,
    gen,
    opt_critic,
    opt_gen,
    dataset,
    wandb,
    start_epoch=0,
    best_fid=float("inf"),
    best_fid_epoch=0,
    n_critic=5,
    clip_value=0.01,
):
    """
    WGAN training loop with weight clipping
    """
    fixed_noise = torch.randn((cfg.batch_size, cfg.z_dim)).to(cfg.device)

    # Define master timeline metric
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")

    # Setup CSV logging
    train_csv_path = utils.setup_training_csv("wgan")
    val_csv_path = utils.setup_training_csv("wgan")  # Using same CSV for simplicity

    # Create data loader for efficient batching
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

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        process = psutil.Process(os.getpid())
        print(
            f"Epoch: {epoch} | RAM GB: {process.memory_info().rss / 1024**3:.2f} \
              | GPU GB: {torch.cuda.memory_allocated() / 1024**3:.2f}"
        )

        # WGAN-specific: train critic more than generator
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(cfg.device)
            batch_size = real.shape[0]

            ### Train Critic: max E[critic(real)] - E[critic(fake)] ###
            for _ in range(n_critic):
                noise = torch.randn((batch_size, cfg.z_dim)).to(cfg.device)
                fake = gen(noise)
                critic_real = critic(real)
                critic_fake = critic(fake)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            ### Train Generator: max E[critic(gen(fake))] ###
            noise = torch.randn((batch_size, cfg.z_dim)).to(cfg.device)
            fake = gen(noise)
            critic_fake = critic(fake)
            loss_gen = -torch.mean(critic_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # --- LOG LOSSES EVERY EPOCH ---
        wandb.log(
            {
                "Loss/Critic": loss_critic.item(),
                "Loss/Generator": loss_gen.item(),
                "epoch": epoch,
            },
            commit=False,
        )

        # --- VISUALS AT START OF EPOCH ---
        if batch_idx == 0 and (epoch == start_epoch or epoch % 10 == 0):
            print(
                f"Epoch [{epoch}/{cfg.num_epochs}] Loss C: {loss_critic.item():.4f}, Loss G: {loss_gen.item():.4f}"
            )
            utils.log_tensorboard_visuals(wandb, gen, real, fixed_noise, epoch)

        # --- FID CALCULATION AT END OF EPOCH ---
        if (epoch % cfg.fid_interval == 0) or (epoch == cfg.num_epochs - 1):
            utils.save_model(
                gen, critic, opt_gen, opt_critic, epoch, filename="latest_gan.pth"
            )
            current_fid = metrics.compute_fid_from_images(gen, dataset, FrechetInceptionDistance(feature=cfg.fid_dims, normalize=True).to(cfg.device))
            # Note: compute_fid_from_images expects a loader, but we can adapt or use alternative

            # VERIFIED FIX: Bundle 'epoch' into the dictionary and commit the full row to WandB
            wandb.log({"Metrics/FID": current_fid, "epoch": epoch}, commit=True)
            print(f"--- Epoch [{epoch}] FID Score: {current_fid:.4f} ---")

            # Log losses and FID to training CSV
            utils.log_training_row(train_csv_path, epoch,
                                   loss_g=loss_gen.item(),
                                   loss_d=loss_critic.item(),
                                   fid_train=current_fid)
            # Print CSV row for output log
            print(f"Epoch [{epoch}/{cfg.num_epochs}] CSV: G_loss={loss_gen.item():.6f}, D_loss={loss_critic.item():.6f}, FID_train={current_fid:.6f}")

            # Checkpoint: Save as 'best' if quality improved
            if current_fid < best_fid:
                best_fid = current_fid
                best_fid_epoch = epoch
                # --- NEW: Update WandB summary immediately for interruption resilience ---
                wandb.summary["Best FID Epoch"] = best_fid_epoch
                wandb.summary["Best FID Score"] = best_fid
                utils.save_model(
                    gen, critic, opt_gen, opt_critic, epoch, filename="best_gan.pth"
                )
        else:
            # VERIFIED FIX: Commits the losses and moves the custom timeline forward on non-FID epochs
            wandb.log({"epoch": epoch}, commit=True)
            # Log losses to training CSV (no FID)
            utils.log_training_row(train_csv_path, epoch,
                                   loss_g=loss_gen.item(),
                                   loss_d=loss_critic.item(),
                                   fid_train=None)
            # Print CSV row for output log
            print(f"Epoch [{epoch}/{cfg.num_epochs}] CSV: G_loss={loss_gen.item():.6f}, D_loss={loss_critic.item():.6f}, FID_train=N/A")

        # End of Epoch cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # --- FINAL RUN SUMMARY ---
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid

    return opt_critic, opt_gen, best_fid, best_fid_epoch, val_csv_path


def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("wgan-model", type="model")
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    file_path = f"{gan_checkpoints_dir}/best_gan.pth"
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def createWandB():
    wandb.init(
        project="super-gans-project",
        name="WGAN",
        config={
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "z_dim": cfg.z_dim,
            "image_size": cfg.image_size,
            "num_channels": cfg.num_channels,
            "n_critic": cfg.n_critic,
            "weight_clip": cfg.weight_clip,
        },
    )
    return wandb


def main(restart=False, best_fid=float("inf"), best_fid_epoch=0):
    # writer = SummaryWriter("logs/wgan_run_1")
    wandb = createWandB()
    utils.prepare_data()
    train_dataset = utils.load_data()

    critic = Critic(image_size=cfg.image_size).to(cfg.device)
    gen = Generator(image_size=cfg.image_size).to(cfg.device)
    # Apply DCGAN weight initialization
    critic.apply(initialize_weights)
    gen.apply(initialize_weights)
    # Compile models for faster training (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        critic = torch.compile(critic)
        gen = torch.compile(gen)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.lr, betas=(0.5, 0.9))
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr, betas=(0.5, 0.9))

    start_epoch = 0
    if restart:
        # Check for existing checkpoint to resume training
        start_epoch = utils.reload_checkpoint_model(gen, critic, opt_gen, opt_critic)

    if start_epoch == 0:
        print("Starting training from scratch")
    else:
        print(f"Resuming training from epoch {start_epoch}")

    opt_critic, opt_gen, best_fid, best_fid_epoch, val_csv_path = training_loop(
        critic, gen, opt_critic, opt_gen, train_dataset, wandb, start_epoch=start_epoch
    )
    utils.save_model(
        gen, critic, opt_gen, opt_critic, f"epoch:{cfg.num_epochs}", "wgan_checkpoint.pth"
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