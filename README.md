## super_gans

# mypi
uv run mypy src/

# uv build
uv add [dependency}]
uv sync --force-reinstall 
uv clean
uv build
uv run python -m super_gans.gans.simple_gan

