# super_gans
This project compares different GAN architectures to augment X-ray datasets with synthetic images, ultimately improving pneumonia detection accuracy.

## GAN Architectures explored
1. Simple GAN
2. DC GAN
3. WGAN
4. WGAN-GP
5. SR GAN
6. Diffusion GAN...

## Implementation
Implementation is based on the reference on [Alladin Persson's GAN models](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs)

## Metrics
1. Frechet Inception Distance (FID)
2. Kernel Inception Distance (KID)


## DataSets:
Supports two datasets. Specify DATA_SOURCE in config.py 
DATA_SOURCE=ptmooney | rsna


1. [Paul Timothy Monny](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
*(For GAN training, we use only the pneumonia subset)*

- Overall Dataset:
    - Total images: 5,856 (5,216 train + 16 val + 624 test)
    - Normal images: 1,583 (27.0%)
    - Pneumonia images: 4,273 (73.0%)


2. [RSNA Pneumonia Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)

- Overall Dataset:
    - Total images: 26,684 
    - Normal images: 20,672 (77.5%)
    - Pneumonia images: 6,012 (22.5%)



# Development Instructions

## uv build
- uv add [dependency]
- uv sync --force-reinstall 
- uv clean
- uv build
- uv run python -m super_gans.gans.simple_gan
- uv run mypy src/


Cleaned up workflow to be used on Intel Mac
1. Clean slate (if things are completely broken) 
    - rm -rf .venv 
    - conda deactivate && conda remove --name super_gans --all -y
2. Recreate the Conda Environment 
    - conda create -n super_gans python=3.12 -y 
    - conda activate super_gans 
3. Install the tricky Intel Mac binaries via Conda-forge 
    - conda install pytorch torchvision torchmetrics -c conda-forge -y 
4. Install your project and remaining dependencies into Conda via uv 
    - uv pip install -e ".[image]" --system 
5. Run your code using the ambient Conda python 
    - PYTHONPATH=src python -m super_gans.gans.simple_gan


## Running GANs:
- super_gans.gans.simple_gan
- super_gans.gans.dcgan
- super_gans.gans.wgan
- super_gans.gans.wgan_gp
- super_gans.gans.sr_gan


## Resources:
- GAN Reference Impl
    - https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs

- Data:
    - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
    - https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

- Metrics Impl:
    - https://github.com/mseitzer/pytorch-fid
    - https://github.com/Lightning-AI/torchmetrics
    - https://torchmetrics.readthedocs.io/