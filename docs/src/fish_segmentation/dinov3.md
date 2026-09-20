# fish_segmentation.dinov3

## Purpose

DINOv3 anomaly-based object detection. Self-supervised trick: dense ViT patch
tokens are compared against each other, patches that differ most from the average
texture of the frame (birds, dolphins, boats on water) get high anomaly scores.
The resulting heatmap → binary mask → connected components yield salient regions
with a **guaranteed-interior point** (the "pole of inaccessibility") per region —
meant as point prompts for SAM3.

Driver: `notebooks/03_dinov3_detect.ipynb`.

## Pipeline stages

1. Sample frames — `fish_segmentation.video_io.load_sampled_frames()`.
2. Preprocess: `remove_letterbox(image, threshold=0.04)` crops black bars;
   `prepare_scaled_tensor(image, processor, scale=4.0, patch_size=16, device)`
   resizes to a patch-multiple resolution (4x by default) and returns
   `(pixel_values, (h_patches, w_patches), (orig_w, orig_h))`.
3. Tokens: `extract_patch_tokens(model, pixel_values, num_patches)` — forward
   pass, drop leading (CLS/registers) tokens, L2-normalize.
4. Heatmap: `compute_anomaly_heatmap(tokens, grid_shape, target_shape,
   max_sample_pool=1500, border_margin_pct=0.05, normalize=True)` — 1 − mean cosine
   similarity per token (a fixed-seed subsample of the reference pool keeps
   O(N·1500) instead of O(N²) and makes runs reproducible), border tokens zeroed
   (ViT border artifact), bicubic upsample to pixels, min-max normalized.
   `normalize=False` returns raw scores, which tiled stitching uses.
5. Mask: `segment_anomalies(norm_score, percentile_threshold=99.5,
   adaptive_block_size=21, morph_kernel_size=3)` — top-tail percentile mask
   ANDed with an adaptive-threshold mask, then morphological opening.
6. Regions: `extract_salient_regions(binary_mask, method="otsu"|"iqr",
   iqr_k=1.5, min_absolute_pixels=20)` — connected components (8-connectivity),
   area thresholding (Otsu on log-areas, or IQR outlier rule), and per region:
   `{'label', 'area', 'center': (x, y), 'max_inscribed_radius'}` where `center`
   is the distance-transform maximum (guaranteed inside the component).
7. End-to-end wrapper: `run_dino_detector(raw_image, model, processor,
    resolution_scale=4.0, percentile_threshold=99.5, visualize=True, device=None)`
    → `(final_mask, norm_score, cropped_image)`.
8. Zoom driver: `run_zoom_detector(..., long_side=1024,
   resolution_scale=4.0)` applies the multiplier to every coarse and recursive
   view. Views exceeding the patch budget are split into overlapping tiles and
   stitched back into native coordinates: each tile's repeating positional bias
   is estimated by the cross-tile phase mean and subtracted, tiles are blended
   with a cosine feather, and the result is min-max normalized once.
   `run_dino_view()` exposes one such view for heatmap diagnostics.
   Passing `trace=[]` collects one record per view, in depth-first order, with
   `depth`, `origin`, `size`, `scale`, `feed_native`, `norm_score`, `mask`,
   `regions`, `zones` and raw `points` — used by the layer-by-layer notebook.
   `merge_gap` and `min_zone_fraction` (defaults 0 = legacy per-component crops)
   shape the zoom context: boxes closer than the gap fuse into one zone and every
   zone grows to at least that fraction of the view's long side.

## Visualization helpers

- `plot_detection_results(image, norm_score, final_mask)` — side-by-side input,
  raw normalized anomaly heatmap, and thresholded-mask overlay for diagnosing
  empty detections.
- `plot_regions_with_centers(image, filtered_mask, regions_info, show_mask=True,
  show_radius=True, ...)` — interior points, optional inscribed circles.
- `plot_zoom_overview(image, trace, detections=None)` — full frame with one
  depth-colored box per traced zoom view, plus merged detections.
- `plot_zoom_level(image, entry)` — one traced view as four image panels: input
  crop, anomaly heatmap, mask + regions + zones, and emitted points.

## Gotchas

- `resolution_scale=4.0` multiplies the zoom detector's capped view resolution;
  it creates 16x as many patches before tiling. `max_patches=4096` keeps each
  tile bounded for the 24 GB 4090. Lower the multiplier if runtime is too high.
- Tiled heatmaps no longer show the tile grid: each tile is an independent image
  for the ViT and carries a positional bias (bright borders, dark center) that
  used to repeat on the lattice. `_tiled_dino_heatmap` estimates that repeating
  component as the cross-tile phase mean (mean raw score per within-tile
  position) and subtracts it per tile — no extra model passes — then feathers
  the overlap. A perfectly flat stitch returns an all-zero map instead of
  amplifying numerical noise.
- `center` in `regions_info` is `(x, y)` (column, row) — the OpenCV convention,
  not numpy `(row, col)`; keep this straight when converting to relative point
  prompts for SAM3.
- Otsu area filtering (`method="otsu"`) needs >2 candidate components; with fewer
  it falls back to the median area. `"iqr"` suits scenes where the mask is nearly
  all noise with a few isolated blobs.
- `trace=` keeps every view's heatmap and mask in RAM (the full-frame root view is
  tens of MB at `resolution_scale=4.0`) — trace one frame at a time.
- Default `merge_gap=0`/`min_zone_fraction=0` reproduces the original behavior:
  each mask component becomes its own tight crop, so border specks turn into
  near-empty sliver views. Set e.g. `merge_gap=0.08, min_zone_fraction=0.25` so
  every zoom pass carries context and groups several nearby detections.
- Enlarged/merged zones are still dropped when they exceed `max_zone_fraction`
  (0.8) of the view, so a chain of border blobs spanning the frame yields no zoom
  rather than a useless full-frame pass.
- The backbone `facebook/dinov3-vitl16-pretrain-lvd1689m` is a gated HF repo —
  `load_env()` first.
