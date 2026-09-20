# notebooks/08_dinov3_heatmap_ab.ipynb

## Purpose

A/B of the DINO detector score and mask knobs (`AnomalyOptions`): heatmap score
mode (`mean`/`knn`) and local contrast, mask method (`adaptive` vs `hysteresis`
with its `low_percentile`), plus the same option sets through the end-to-end
`run_zoom_detector`.

## Process

1. `load_env()` + `ROOT = find_repo_root()`; pick device; `VIDEO_PATH` defaults to
   `data/test_media/dolphin_00.mp4` (uncomment the processed-dataset path for
   real footage).
2. Load backbone + 3 sampled frames; use the middle frame (letterbox-cropped).
3. Heatmap panels: one scaled coarse view per variant (`mean`, `knn5`,
   `mean + lc 0.05`, `mean + lc 0.25`) with mean/p99 annotations.
4. Mask panels: `segment_anomalies` with `adaptive`, `hysteresis` p98/p97 and
   p97 + closing, same `mean` heatmap, each titled with its region count.
5. End-to-end: `run_zoom_detector` for legacy (`mean + adaptive`), the validated
   defaults (`mean + hysteresis p98`) and `lc 0.25 + hysteresis p98`; detections
   plotted by level plus a printed count/radius/level table.

## Inputs / outputs

- Input: a video path (default tracked test media).
- Output: in-memory heatmaps/masks/detections + plots; no files written.

Details: [dinov3.md](../src/fish_segmentation/dinov3.md).
