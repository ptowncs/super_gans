import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import RandomSampler
from PIL import Image  # Import PIL Image
import super_gans.config as cfg
from torchvision.utils import save_image
from pytorch_fid import fid_score


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


def load_data():
    def is_valid_image(filename):
        return not filename.startswith("._")  # skip hidden macOS files

    transforms_pipeline = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=cfg.num_channels
            ),  # Force 1 channel
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),  # image to tensor
            transforms.Normalize((0.5,), (0.5,)),  # normalize images , [0,1] to [-1,1]
        ]
    )
    dataset_path = f"{cfg.DATA_DIR}/datasets/paultimothymooney/chest-xray-pneumonia"
    chest_xray_ds = f"{dataset_path}/versions/2/chest_xray"
    dataset = datasets.ImageFolder(
        root=f"{chest_xray_ds}/train",
        transform=transforms_pipeline,
        is_valid_file=is_valid_image,
    )
    return dataset


def training_loop(disc, gen, dataset):
    fixed_noise = torch.randn((cfg.batch_size, cfg.z_dim)).to(cfg.device)

    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr)
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()

    writer_fake = SummaryWriter("logs/fake")
    writer_real = SummaryWriter("logs/real")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    step = 0

    for epoch in range(cfg.num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, cfg.image_dim).to(
                cfg.device
            )  # Flatten image into vectors
            batch_size = real.shape[0]

            # Train Discriminator
            noise = torch.randn(batch_size, cfg.z_dim).to(
                cfg.device
            )  # generator makes fakes
            fake = gen(noise)

            disc_real = disc(real).view(-1)  # disc tries to classify real images
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(fake.detach()).view(
                -1
            )  # disc tries to classify real images
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2

            # Backpropagation
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(
                -1
            )  # Generator looks at images that were detected as fake
            lossG = criterion(output, torch.ones_like(output))

            # update generator weights
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
            fixed_noise = torch.randn((cfg.batch_size, cfg.z_dim)).to(cfg.device)
            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{cfg.num_epochs}] Batch {batch_idx}/{len(loader)} "
                    f"Loss D: {lossD.item():.4f}, loss G: {lossG.item():.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(
                        -1, cfg.num_channels, cfg.image_size, cfg.image_size
                    )
                    data = real.reshape(
                        -1, cfg.num_channels, cfg.image_size, cfg.image_size
                    )

                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image(
                        "Pneumonia Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Pneumonia Real Images", img_grid_real, global_step=step
                    )

                    step += 1
    return opt_disc, opt_gen


def save_model(disc, gen, opt_disc, opt_gen):
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    gan_results_dir = f"{cfg.RESULTS_DIR}/generated_images_fid"
    os.makedirs(gan_checkpoints_dir, exist_ok=True)
    os.makedirs(gan_results_dir, exist_ok=True)

    checkpoint = {
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_G_state_dict": opt_gen.state_dict(),
        "optimizer_D_state_dict": opt_disc.state_dict(),
    }
    print("Saving GAN state to GAN Checkpoints")
    torch.save(checkpoint, f"{gan_checkpoints_dir}/simple_gan_checkpoint.pth")


def save_images_fid(dataset, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    for i in range(min(cfg.num_real_images_to_save, len(dataset))):
        image, _ = dataset[i]  # image is a tensor in shape (1, 64, 64)

        # Convert grayscale -> RGB by repeating channels
        image_rgb = image.repeat(3, 1, 1)

        torchvision.utils.save_image(
            image_rgb,
            os.path.join(to_dir, f"pneumonia_{i:04d}.png"),
            normalize=False,
            value_range=(-1, 1),
        )

    print(
        f"Saved {min(cfg.num_real_images_to_save, len(dataset))} real Pneumonia images to {to_dir}/"
    )
    return real_images_dir


def reloadModel():
    gan_checkpoints_dir = f"{cfg.MODELS_DIR}/gan_checkpoints"
    #  Instantiate models
    generator = Generator().to(cfg.device)
    discriminator = Discriminator().to(
        cfg.device
    )  # Discriminator isn't strictly needed for FID, but useful if you need to inspect it.

    #  Path to the checkpoint file
    checkpoint_path = f"{gan_checkpoints_dir}/simple_gan_checkpoint.pth"

    #  Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        # discriminator.load_state_dict(checkpoint["discriminator_state_dict"]) # Optional
        print(f"Checkpoint loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        # Handle the error, maybe exit or use default models
    except KeyError as e:
        print(
            f"Error loading state_dict: Missing key {e}. Ensure checkpoint structure matches."
        )
        # Handle error if checkpoint dictionary keys don't match.

    #  Set generator to evaluation mode
    generator.eval()
    return generator


def generate_images_fid(generator, generated_images_dir):
    # Generate random noise vectors
    gen_noise = torch.randn(cfg.num_images_to_generate, cfg.z_dim).to(cfg.device)

    # Generate images
    with torch.no_grad():
        generated_images = generator(gen_noise).reshape(
            -1, cfg.num_channels, cfg.image_size, cfg.image_size
        )  # Reshape for saving/display

    os.makedirs(generated_images_dir, exist_ok=True)

    # Iterate through the batch and save each image separately
    for i, image in enumerate(generated_images):
        # Construct the filename for each image
        filename = os.path.join(
            generated_images_dir, f"generated_image_{i:04d}.png"
        )  # Using f-strings for formatted filename

        # Save the individual image (image tensor has shape (channels, height, width))
        save_image(image, filename, normalize=False)

    print(
        f"{cfg.num_images_to_generate} individual images saved to {generated_images_dir}/"
    )


def calc_fid_score(real_images_dir, generated_images_dir):
    # Paths to your image directories
    print(f"Path being checked:{repr(real_images_dir)}")
    if not os.path.exists(real_images_dir):
        raise ValueError(f"Real image path not found: {real_images_dir}")
    if not os.path.exists(generated_images_dir):
        raise ValueError(f"Generated image path not found: {generated_images_dir}")

    # (Make sure to populate `real_images` with your actual dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Calculate FID
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_dir, generated_images_dir],
        batch_size=50,  # Adjust batch size based on available GPU memory
        device=device,
        dims=2048,  # Inception v3 output dimension
    )
    return fid_value


if __name__ == '__main__':
    dataset = load_data()
    disc = Discriminator().to(cfg.device)
    gen = Generator().to(cfg.device)
    opt_disc, opt_gen = training_loop(disc, gen, dataset)
    save_model(disc, gen, opt_disc, opt_gen)
    real_images_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
    generated_images_dir = f"{cfg.RESULTS_DIR}/fake_images_fid"
    save_images_fid(dataset, real_images_dir)
    generate_images_fid(reloadModel(), generated_images_dir)
    fid_value = calc_fid_score(real_images_dir, generated_images_dir)
    print(f"FID score: {fid_value}")
