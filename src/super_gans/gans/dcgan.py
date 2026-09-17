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


class Discriminator(nn.Module):
    """
    Discriminator: increases the number of channels(features) while downsampling the spatial dimensions to half
    Skips every other pixel and looks at window of 4 x 4 to create 1 pixel
    """

    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                cfg.num_channels, 32, 4, 2, 1
            ),  # num_channels , 128x128 -> 32, 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32, 64 x 64 -> 64,  32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64,  32 x 32 -> 128, 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 128, 16 x 16 -> 256, 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
        )

    def forward(self, x):
        return self.disc(x)  # Defines how data flows through the network


class Generator(nn.Module):
    """
    Generator: decreases the number of features to num_channels while upsampling spatial dimensions by doubling.
    It first inserts 0s in alternate rows and columns and skips 1 pixel to double the image size
    """

    def __init__(self):
        super().__init__()
        # z_dim is size of noise vector , image_dim is total pixels in image
        self.gen = nn.Sequential(
            nn.Linear(cfg.z_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, cfg.num_channels, 4, 2, 1),  # 128
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


def training_loop(
    disc,
    gen,
    opt_disc,
    opt_gen,
    dataset,
    wandb,
    start_epoch=0,
    best_fid=float("inf"),
    best_fid_epoch=0,
):
    fixed_noise = torch.randn((cfg.batch_size, cfg.z_dim)).to(cfg.device)

    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    # feature=64 uses a lower layer of Inception; it's faster for monitoring
    fid_metric = FrechetInceptionDistance(feature=cfg.fid_dims, normalize=True).to(
        cfg.device
    )

    # Define master timeline metric
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")

    for epoch in range(start_epoch, cfg.num_epochs):
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

        # --- VISUALS AT START OF EPOCH ---
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}/{cfg.num_epochs}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}"
            )
            utils.log_tensorboard_visuals(wandb, gen, real_orig, fixed_noise, epoch)

        # --- FID CALCULATION AT END OF EPOCH ---
        if (epoch % cfg.fid_interval == 0) or (epoch == cfg.num_epochs - 1):
            utils.save_model(
                gen, disc, opt_gen, opt_disc, epoch, filename="latest_gan.pth"
            )
            current_fid = metrics.compute_fid_from_images(gen, loader, fid_metric)
            fid_metric.reset()

            # VERIFIED FIX: Bundle 'epoch' into the dictionary and commit the full row to WandB
            wandb.log({"Metrics/FID": current_fid, "epoch": epoch}, commit=True)
            print(f"--- Epoch [{epoch}] FID Score: {current_fid:.4f} ---")

            # Checkpoint: Save as 'best' if quality improved
            if current_fid < best_fid:
                best_fid = current_fid
                best_fid_epoch = epoch
                utils.save_model(
                    gen, disc, opt_gen, opt_disc, epoch, filename="best_gan.pth"
                )
        else:
            # VERIFIED FIX: Commits the losses and moves the custom timeline forward on non-FID epochs
            wandb.log({"epoch": epoch}, commit=True)

        # End of Epoch cleanup
        torch.cuda.empty_cache()
        gc.collect()

    # --- FINAL RUN SUMMARY ---
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid

    return opt_disc, opt_gen


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
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr, betas=cfg.betas)

    start_epoch = 0
    if restart:
        # Check for existing checkpoint to resume training
        start_epoch = utils.reload_checkpoint_model(gen, disc, opt_gen, opt_disc)

    if start_epoch == 0:
        print("Starting training from scratch")
    else:
        print(f"Resuming training from epoch {start_epoch}")

    training_loop(
        disc, gen, opt_disc, opt_gen, train_dataset, wandb, start_epoch=start_epoch
    )
    utils.save_model(
        gen, disc, opt_gen, opt_disc, f"epoch:{cfg.num_epochs}", "dc_gan_checkpoint.pth"
    )

    real_images_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
    generated_images_dir = f"{cfg.RESULTS_DIR}/fake_images_fid"
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
    wandb.finish()


if __name__ == "__main__":
    main()
