## super_gans

# mypi
uv run mypy src/

# uv build
uv add [dependency}]
uv sync --force-reinstall 
uv clean
uv build
uv run python -m super_gans.gans.simple_gan


Cleaned up workflow to be used on Intel Mac
# 1. Clean slate (if things are completely broken) 
rm -rf .venv 
conda deactivate conda remove --name super_gans --all -y # 
2. Recreate the Conda Environment 
conda create -n super_gans python=3.12 -y 
conda activate super_gans 
# 3. Install the tricky Intel Mac binaries via Conda-forge 
conda install pytorch torchvision torchmetrics -c conda-forge -y 
# 4. Install your project and remaining dependencies into Conda via uv 
uv pip install -e ".[image]" --system 
# 5. Run your code using the ambient Conda python 
PYTHONPATH=src python -m super_gans.gans.simple_gan


Resources:
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs

Data:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

Metrics:
https://github.com/mseitzer/pytorch-fid
https://github.com/Lightning-AI/torchmetrics
https://torchmetrics.readthedocs.io/
