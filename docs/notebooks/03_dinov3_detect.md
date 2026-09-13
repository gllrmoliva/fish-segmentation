# notebooks/03_dinov3_detect.ipynb

## Purpose

DINOv3 anomaly detection driver: heatmap → binary mask → salient regions with
guaranteed-interior points (candidate point prompts for SAM3).

## Process

1. `load_env()` + `ROOT = find_repo_root()`; pick device.
2. `load_backbone("facebook/dinov3-vitl16-pretrain-lvd1689m", device)` (gated HF repo — needs `HF_TOKEN`).
3. `load_sampled_frames(data/test_media/dolphin_00.mp4, n_frames=5)`.
4. Per frame: `run_dino_detector(..., resolution_scale=4.0)` +
   `extract_salient_regions(..., method="otsu", min_absolute_pixels=30)`;
   prints the average regions-per-frame count.
5. `plot_regions_with_centers(...)` on one result.

## Inputs / outputs

- Input: a video path (default tracked test media).
- Output: in-memory `results` list of `(cropped_img, clean_mask, regions)` + plots.

Details: [dinov3.md](../src/fish_segmentation/dinov3.md).
