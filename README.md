# Fish Segmentation
This project uses [sam3](https://github.com/facebookresearch/sam3).

## Requirements

- Git
- [uv](https://docs.astral.sh/uv/)
- Python 3.12
- ffmpeg
- exiftool

## Installation

From the repository root, sync the project environment:

```bash
uv sync
```

> If Python 3.12 is not installed, let `uv` install it first:
```bash
uv python install 3.12
uv sync
```

## Run JupyterLab

Start JupyterLab from the repository root with:

```bash
uv run jupyter lab
```

## Create a kernel

If you have jupyter lab already installed, create a new kernel with:

```bash
uv run python -m ipykernel install --user --name=fish-segmentation --display-name="smell-like-fish"
```

