# notebooks/07_dinov3_zoom_levels.ipynb

## Purpose

Layer-by-layer diagnostics of the coarse-to-fine zoom detector: every DINO view
(= layer) of `run_zoom_detector` is rendered as images instead of the pipeline
only returning merged points.

## Process

1. `load_env()` + `ROOT = find_repo_root()`; pick device; same constants as 03
   (`long_side=1024`, `resolution_scale=4.0`).
2. `load_backbone("facebook/dinov3-vitl16-pretrain-lvd1689m", device)`.
3. Load the processed video, falling back to `data/test_media/dolphin_00.mp4`;
   use the middle of 3 sampled frames.
4. `levels = []`; `run_zoom_detector(..., trace=levels)` with
   `merge_gap=0.08, min_zone_fraction=0.25` (context crops: fuse nearby boxes,
   never zoom into a crop smaller than 25% of the view's long side); print views
   per depth and the merged-detection count. Detections are deepest-wins: they
   carry `level`, `n_levels`/`levels` (confirming depths), `bbox` and region
   metadata (`area`, `solidity`, `score_mean`).
5. `plot_zoom_overview(frame, levels, detections)` — original frame with one
   depth-colored box per traced view, plus the merged detections and their boxes.
6. Loop over `levels` → `plot_zoom_level(frame, entry)`: four image panels
   per layer (input crop; anomaly heatmap; mask + regions + zones; emitted
   `RawPoint`s).
7. Plot the final merged detections colored by confirmation level.

## Inputs / outputs

- Input: a video path (default processed dataset video, tracked-test-media fallback).
- Output: in-memory `levels` trace + `detections`; plots only (nothing written).

Details: [dinov3.md](../src/fish_segmentation/dinov3.md).
