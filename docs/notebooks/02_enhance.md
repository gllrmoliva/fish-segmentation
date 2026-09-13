# notebooks/02_enhance.ipynb

## Purpose

Marine video enhancement driver: glint/foam inpainting, edge-preserving
smoothing, CLAHE → clean H.264 MP4 for the downstream SAM3/DINOv3 notebooks.

## Process

1. `load_env()` + `ROOT = find_repo_root()`; ensures `notebooks/outputs/` exists.
2. `process_marine_video_ffmpeg(<input>, notebooks/outputs/preprocessed.mp4, fps=15.0)`.
3. Side-by-side `IPython.display.Video` of input vs output.

## Inputs / outputs

- Input: `data/test_media/dolphin_00.mp4` (swap for any ingested
  `dataset/processed/<folder>/video.mp4`).
- Output: `notebooks/outputs/preprocessed.mp4` (gitignored).

Details: [enhance.md](../src/fish_segmentation/enhance.md).
