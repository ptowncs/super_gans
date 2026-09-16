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
from torchvision.utils import save_image

import super_gans.config as cfg
import wandb
from super_gans import utils
from super_gans import metrics


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

    criterion = nn.BCELoss()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    # feature=64 uses a lower layer of Inception; it's faster for monitoring
    fid_metric = FrechetInceptionDistance(feature=cfg.fid_dims, normalize=True).to(
        cfg.device
    )

    # Define master timeline metric
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")

    for epoch in range(cfg.num_epochs):
        process = psutil.Process(os.getpid())
        print(
            f"Epoch: {epoch} | RAM GB: {process.memory_info().rss / 1024**3:.2f} \
              | GPU GB: {torch.cuda.memory_allocated() / 1024**3:.2f}"
        )

        for batch_idx, (real_orig, _) in enumerate(loader):
            real = real_orig.view(-1, cfg.image_dim).to(cfg.device)
            batch_size = real.shape[0]

            ### Train Discriminator ###
            disc.zero_grad(set_to_none=True)

            # Pass 1: Discriminator updates on Real Images
            disc_real = disc(real).view(-1)

            # Apply One-Sided Label Smoothing to keep gradients healthy
            # lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            lossD_real = criterion(disc_real, torch.ones_like(disc_real) * 0.9)

            # Pass 2: Discriminator updates on Fake Images
            noise = torch.randn(batch_size, cfg.z_dim).to(cfg.device)
            fake = gen(noise)
            # Explicitly detach the fake images so backpropagation doesn't leak into the Generator
            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2
            lossD.backward()  # NO retain_graph needed anymore!
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
        # writer.flush()
        torch.cuda.empty_cache()
        gc.collect()

    # --- FINAL RUN SUMMARY ---
    wandb.summary["Best FID Epoch"] = best_fid_epoch
    wandb.summary["Best FID Score"] = best_fid

    return opt_disc, opt_gen


def uploadLogsAndMetricsToWandB(wandb):
    # Upload model
    artifact = wandb.Artifact("simple-gan-model", type="model")
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    file_path = f"{gan_checkpoints_dir}/best_gan.pth"
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)


def createWandB():
    wandb.init(
        project="super-gans-project",
        name="Simple_GAN",
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
    # writer = SummaryWriter("logs/simple_gan_run_1")
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
        disc,
        gen,
        opt_disc,
        opt_gen,
        train_dataset,
        wandb,
        start_epoch,
        best_fid,
        best_fid_epoch,
    )
    utils.save_model(
        gen,
        disc,
        opt_gen,
        opt_disc,
        f"epoch:{cfg.num_epochs}",
        "simple_gan_checkpoint.pth",
    )

    real_images_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
    generated_images_dir = f"{cfg.RESULTS_DIR}/fake_images_fid"
    fid_dataset = utils.build_fid_evaluation_dataset(target_samples=cfg.num_images_fid_score)
    utils.save_images_fid(fid_dataset, real_images_dir)
    # Need to reload just the generator for final evaluation - always load best model
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    best_model_path = f"{gan_checkpoints_dir}/best_gan.pth"
    utils.load_best_model(gen, best_model_path)
    utils.generate_images_fid(gen, generated_images_dir)
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
