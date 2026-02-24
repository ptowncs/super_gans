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
from super_gans.utils import get_dataset_path
from torchvision.utils import save_image
from pytorch_fid import fid_score
from torchmetrics.image.fid import FrechetInceptionDistance


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
    dataset_path = get_dataset_path("paultimothymooney/chest-xray-pneumonia/versions/2")
    chest_xray_ds =  f"{dataset_path}/chest_xray"
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
    writer = SummaryWriter("logs/simple_gan_run_1")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    # 'step' tracks total batches seen (X-axis for loss charts)
    step = 0
    # feature=64 uses a lower layer of Inception; it's faster for monitoring
    fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(cfg.device)
    best_fid = float('inf') # Initialize with infinity

    for epoch in range(cfg.num_epochs):
        for batch_idx, (real_orig, _) in enumerate(loader):
            real = real_orig.view(-1, cfg.image_dim).to(cfg.device)
            batch_size = real.shape[0]

            ### Train Discriminator ###
            noise = torch.randn(batch_size, cfg.z_dim).to(cfg.device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            # Backpropagation
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            ### Train Generator ###
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            # --- LOG LOSSES EVERY BATCH ---
            writer.add_scalar("Loss/Discriminator", lossD.item(), global_step=step)
            writer.add_scalar("Loss/Generator", lossG.item(), global_step=step)
            step += 1 # Increment every batch for smooth loss curves

            # --- VISUALS AT START OF EPOCH ---
            if batch_idx == 0:
                print(f"Epoch [{epoch}/{cfg.num_epochs}] Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")
                log_tensorboard_visuals(writer, gen, real_orig, fixed_noise, epoch)

        # --- FID CALCULATION AT END OF EPOCH ---
        if epoch % cfg.fid_interval == 0 or epoch == cfg.num_epochs:
            current_fid = calculate_fid(gen, loader, fid_metric)
            writer.add_scalar("Metrics/FID", current_fid, global_step=epoch)
            print(f"--- Epoch [{epoch}] FID Score: {current_fid:.4f} ---")
        writer.flush()
        # Checkpoint: Save as 'best' if quality improved
        if current_fid < best_fid:
            best_fid = current_fid
            save_model(gen, disc, opt_gen, opt_disc, epoch, filename="best_gan.pth")
        # Always save 'latest' in case Kaggle session times out
        save_model(gen, disc, opt_gen, opt_disc, epoch, filename="latest_gan.pth")
    
    writer.flush()
    writer.close()
    return opt_disc, opt_gen

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

def save_images_fid(dataset, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    for i in range(min(cfg.num_images_fid_score, len(dataset))):
        image, _ = dataset[i]  # image is a tensor in shape (1, 64, 64)

        # Convert grayscale -> RGB by repeating channels
        image_rgb = image.repeat(3, 1, 1)
        # Generator uses Tanh (outputting [-1, 1]), ensure normalize=True and value_range=(-1, 1) 
        # so the PNGs are stored as standard [0, 255] pixel values correctly.
        torchvision.utils.save_image(
            image_rgb,
            os.path.join(to_dir, f"pneumonia_{i:04d}.png"),
            normalize=True,
            value_range=(-1, 1),
        )

    print(
        f"Saved {min(cfg.num_images_fid_score, len(dataset))} real Pneumonia images to {to_dir}/"
    )

def generate_images_fid(generator, generated_images_dir):
    # Generate random noise vectors
    gen_noise = torch.randn(cfg.num_images_fid_score, cfg.z_dim).to(cfg.device)

    # Generate images
    with torch.no_grad():
        generated_images = generator(gen_noise).reshape(
            -1, cfg.num_channels, cfg.image_size, cfg.image_size
        )  # Reshape for saving/display

    os.makedirs(generated_images_dir, exist_ok=True)

    # Iterate through the batch and save each image separately
    for i, image in enumerate(generated_images):
        # Convert Grayscale -> RGB to match the real images directory
        image_rgb = image.repeat(3, 1, 1)
        # Construct the filename for each image
        filename = os.path.join(
            generated_images_dir, f"generated_image_{i:04d}.png"
        )  # Using f-strings for formatted filename

        # Save the individual image (image tensor has shape (channels, height, width))
        save_image(image_rgb, filename, normalize=True, value_range=(-1, 1))

    print(
        f"{cfg.num_images_fid_score} individual images saved to {generated_images_dir}/"
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

def calculate_fid(gen, loader, fid_metric):
    """
    Calculates FID score by comparing real images from the loader 
    with generated images from the generator.
    """
    gen.eval()
    fid_metric.reset()
    
    # Calculate how many batches we need to reach num_samples
    batch_size = cfg.batch_size
    n_batches = cfg.num_images_fid_sample // batch_size
    data_iter = iter(loader)

    with torch.no_grad():
        for _ in range(n_batches):
            # --- 1. Process Real Images ---
            try:
                real_batch, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real_batch, _ = next(data_iter)
                
            real_batch = real_batch[:batch_size].to(cfg.device)
            # Convert [1, 64, 64] -> [3, 64, 64] and map [-1, 1] -> [0, 1]
            real_rgb = (real_batch.repeat(1, 3, 1, 1) + 1.0) / 2.0
            fid_metric.update(real_rgb, real=True)

            # --- 2. Process Fake Images ---
            noise = torch.randn(batch_size, cfg.z_dim).to(cfg.device)
            fake_batch = gen(noise).reshape(-1, cfg.num_channels, cfg.image_size, cfg.image_size)
            # Convert [1, 64, 64] -> [3, 64, 64] and map [-1, 1] -> [0, 1]
            fake_rgb = (fake_batch.repeat(1, 3, 1, 1) + 1.0) / 2.0
            fid_metric.update(fake_rgb, real=False)

        # --- 3. Compute and Log ---
        fid_score = fid_metric.compute().item()
        
    gen.train()
    return fid_score

def log_tensorboard_visuals(writer, gen, real_batch, fixed_noise, epoch):
    """
    Captures the current state of generation vs real images.
    """
    gen.eval() 
    with torch.no_grad():
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
        img_grid_fake = torchvision.utils.make_grid(
            fake_rgb, normalize=True, value_range=(-1, 1)
        )
        img_grid_real = torchvision.utils.make_grid(
            real_rgb, normalize=True, value_range=(-1, 1)
        )

        # 5. Log to TensorBoard
        writer.add_image("Images/Generated", img_grid_fake, global_step=epoch)
        writer.add_image("Images/Real", img_grid_real, global_step=epoch)
    
    gen.train()

if __name__ == '__main__':
    dataset = load_data()
    disc = Discriminator().to(cfg.device)
    gen = Generator().to(cfg.device)
    opt_disc, opt_gen = training_loop(disc, gen, dataset)
    save_model(disc, gen, opt_disc, opt_gen, "simple_gan_checkpoint.pth")
    
    real_images_dir = f"{cfg.RESULTS_DIR}/real_images_fid"
    generated_images_dir = f"{cfg.RESULTS_DIR}/fake_images_fid"
    save_images_fid(dataset, real_images_dir)
    last_model = reloadModel("simple_gan_checkpoint.pth")
    #best_model = reloadModel("best_gan.pth")
    generate_images_fid(last_model, generated_images_dir)
    fid_value = calc_fid_score(real_images_dir, generated_images_dir)
    print(f"FID score: {fid_value}")
