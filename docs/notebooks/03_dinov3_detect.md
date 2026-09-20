# notebooks/03_dinov3_detect.ipynb

## Purpose

DINOv3 anomaly detection driver: heatmap → binary mask → salient regions with
guaranteed-interior points (candidate point prompts for SAM3).

## Process

1. `load_env()` + `ROOT = find_repo_root()`; pick device.
2. `load_backbone("facebook/dinov3-vitl16-pretrain-lvd1689m", device)` (gated HF repo — needs `HF_TOKEN`).
3. Load the configured processed video, falling back to
   `data/test_media/dolphin_00.mp4` when it is unavailable.
4. Per frame: `run_zoom_detector(..., long_side=1024,
   resolution_scale=4.0)`; oversized views are tiled automatically and the
   average detections-per-frame count is printed.
5. Run the initial scaled coarse pass once with `run_dino_view()` and display
   the raw DINO heatmap beside the thresholded mask.
6. Plot merged detections on the sampled frames.

## Inputs / outputs

- Input: a video path (default tracked test media).
- Output: in-memory `results` list of `(cropped_img, clean_mask, regions)` + plots.

Details: [dinov3.md](../src/fish_segmentation/dinov3.md).
