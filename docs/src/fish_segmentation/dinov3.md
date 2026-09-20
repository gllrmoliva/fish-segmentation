# fish_segmentation.dinov3

## Purpose

DINOv3 anomaly-based object detection. Self-supervised trick: dense ViT patch
tokens are compared against each other, patches that differ most from the average
texture of the frame (birds, dolphins, boats on water) get high anomaly scores.
The resulting heatmap → binary mask → connected components yield salient regions
with a **guaranteed-interior point** (the "pole of inaccessibility") per region —
meant as point prompts for SAM3.

Driver: `notebooks/03_dinov3_detect.ipynb` (A/B of the score/mask knobs:
`notebooks/08_dinov3_heatmap_ab.ipynb`).

## Options (`AnomalyOptions`)

Frozen dataclass bundling the knobs of the end-to-end detectors; its defaults are
the A/B-validated pipeline. `run_dino_detector`, `run_dino_view` and
`run_zoom_detector` all take `options=`.

- `score_mode`: `"mean"` (default) or `"knn"`.
- `knn_k`: nearest references averaged in `"knn"` mode.
- `local_contrast_strength`: 0 disables (default); >0 high-passes the score map.
- `local_contrast_sigma_pct`: blur sigma as a fraction of the longer map side.
- `mask_method`: `"hysteresis"` (default) or `"adaptive"`.
- `low_percentile`: growth threshold for hysteresis.
- `closing_kernel_size`: optional closing before the final opening.

## Pipeline stages

1. Sample frames — `fish_segmentation.video_io.load_sampled_frames()`.
2. Preprocess: `remove_letterbox(image, threshold=0.04)` crops black bars;
   `prepare_scaled_tensor(image, processor, scale=4.0, patch_size=16, device)`
   resizes to a patch-multiple resolution (4x by default) and returns
   `(pixel_values, (h_patches, w_patches), (orig_w, orig_h))`.
3. Tokens: `extract_patch_tokens(model, pixel_values, num_patches)` — forward
   pass, drop leading (CLS/registers) tokens, L2-normalize.
4. Heatmap: `compute_anomaly_heatmap(tokens, grid_shape, target_shape,
   max_sample_pool=1500, border_margin_pct=0.05, normalize=True,
   score_mode="mean", knn_k=5, local_contrast_strength=0.0,
   local_contrast_sigma_pct=0.05)` — `"mean"` scores 1 − mean cosine similarity
   per token; `"knn"` scores 1 − mean of the top-k similarities (water repeats
   everywhere so it keeps close neighbours, object patches do not). A fixed-seed
   subsample of the reference pool keeps O(N·1500) instead of O(N²) and makes
   runs reproducible. Local contrast subtracts a Gaussian-blurred copy of the
   score map (unsharp/high-pass) to flatten the broad warm halo around peaks; it
   is off by default. Border tokens are then zeroed (ViT border artifact), the
   map is bicubic-upsampled to pixels and min-max normalized. `normalize=False`
   returns raw scores, which tiled stitching uses.
5. Mask: `segment_anomalies(norm_score, percentile_threshold=99.5,
   adaptive_block_size=21, morph_kernel_size=3, method="hysteresis",
   low_percentile=98.0, closing_kernel_size=0)`. `"hysteresis"` (default) seeds
   with the top percentile and keeps every connected component of the
   lower-percentile mask that holds a seed (morphological reconstruction, done
   with `cv2.connectedComponents` so long thin bridges connect exactly), then
   applies the final morphological opening. `"adaptive"` is the legacy
   top-percentile ∩ local adaptive core.
6. Regions: `extract_salient_regions(binary_mask, method="otsu"|"iqr",
   iqr_k=1.5, min_absolute_pixels=20, score_map=None)` — connected components
   (8-connectivity), area thresholding (Otsu on log-areas, or IQR outlier rule),
   and per region: `{'label', 'area', 'center': (x, y), 'max_inscribed_radius',
   'bbox', 'solidity'}` where `center` is the distance-transform maximum
   (guaranteed inside the component), `bbox` is the tight component box and
   `solidity = area / (pi * r^2)` (~1 disk, ~1.27 square, larger when the
   component is elongated or hollow). With `score_map` (the `norm_score` of the
   view) it also adds `score_mean`/`score_p95` inside the component.
7. End-to-end wrapper: `run_dino_detector(raw_image, model, processor,
    resolution_scale=4.0, percentile_threshold=99.5, visualize=True, device=None,
    options=AnomalyOptions())` → `(final_mask, norm_score, cropped_image)`.
8. Zoom driver: `run_zoom_detector(..., long_side=1024,
   resolution_scale=4.0, options=AnomalyOptions())` applies the multiplier to
   every coarse and recursive view. Views exceeding the patch budget are split
   into overlapping tiles and stitched back into native coordinates: each tile's
   repeating positional bias is estimated by the cross-tile phase mean and
   subtracted, tiles are blended with a cosine feather, and the result is min-max
   normalized once. Local contrast is applied to the stitched map after the phase
   mean, never per tile. `run_dino_view()` exposes one such view for heatmap
   diagnostics.
   Passing `trace=[]` collects one record per view, in depth-first order, with
   `depth`, `origin`, `size`, `scale`, `feed_native`, `norm_score`, `mask`,
   `regions`, `zones` and raw `points` (`RawPoint` records) — used by the
   layer-by-layer notebook.
   `merge_gap` and `min_zone_fraction` (defaults 0 = legacy per-component crops)
   shape the zoom context: boxes closer than the gap fuse into one zone and every
   zone grows to at least that fraction of the view's long side.
   `min_confirmations=k` keeps only detections confirmed at k different recursion
   depths (default 1 = keep everything) and `keep_absorbed=True` re-emits
   absorbed coarse representatives for A/B comparisons.

## Merge semantics (deepest-wins)

Raw points from every view are merged by `_merge_points(points,
keep_absorbed=False)`. Same-level points are clustered transitively (one center
inside the other's inscribed circle, conservative), and levels are then resolved
from deep to coarse: a representative matching an already kept deeper cluster
(circles overlap OR one center falls inside the other's region box) is
*absorbed* — it only adds confirmation to `n_levels`/`n_points` — so emitted
centers and radii always belong to a real region of the deepest level and never
to an average across levels. A shallow representative matching no deeper
cluster is kept as its own detection (an object only seen at coarse
resolution). `level` is the deepest confirming region; `n_levels`/`levels`
count the recursion depths that confirmed the object.

`run_zoom_detector` normalizes the output to the ORIGINAL frame (letterbox bars
included, offset from `letterbox_box`) so detections are directly usable as SAM3
point prompts (`rel_coordinates=True`). Each detection dict carries `x`, `y`,
`radius`, `level`, `bbox` (normalized), `area`, `solidity`, `score_mean`,
`score_p95`, `n_levels`, `levels` and `n_points`.

## Visualization helpers

- `plot_detection_results(image, norm_score, final_mask)` — side-by-side input,
  raw normalized anomaly heatmap, and thresholded-mask overlay for diagnosing
  empty detections.
- `plot_regions_with_centers(image, filtered_mask, regions_info, show_mask=True,
  show_radius=True, ...)` — interior points, optional inscribed circles.
- `plot_zoom_overview(image, trace, detections=None)` — original frame (bars
  cropped internally) with one depth-colored box per traced zoom view, plus
  merged detections and their boxes.
- `plot_zoom_level(image, entry)` — one traced view as four image panels: input
  crop, anomaly heatmap, mask + regions + zones, and emitted `RawPoint`s. Takes
  the original frame; the view is reconstructed from `entry['origin']`/`size`.

## Gotchas

- The legacy `method="adaptive"` mask ANDs the top percentile with
  `cv2.adaptiveThreshold(..., C=-2)`, which requires beating the local window
  mean by 2. On large frames a smooth peak never does, the core comes out empty
  and **whole frames yield zero regions** (verified on the 4K processed footage
  at `resolution_scale=1.0`). Hysteresis is percentile-only and has no such
  failure mode.
- `low_percentile` tunes the growth: on the tested 4K frames 98 keeps the mask on
  the animal, 97 grows into water texture at deep zoom levels (false-positive
  point clusters), 99 barely grows beyond the seeds. It only matters when
  `method="hysteresis"`.
- Local contrast sigma: a small sigma (~0.05) removes the mid-scale halo and is
  meant for heatmap inspection / the coarse view; a large sigma (~0.25) only
  removes the very-low-frequency positional fog and survives the zoom recursion
  (objects bigger than the blur are hollowed into rings with a small sigma).
  It is off by default: with hysteresis it did not change the validated
  detections but tripled the runtime in the A/B.
- High-passing each tile separately amplifies per-tile water texture and destroys
  cross-tile comparability (scattered specks after stitching); `_tiled_dino_heatmap`
  therefore applies local contrast once to the stitched map.
- `resolution_scale=4.0` multiplies the zoom detector's capped view resolution;
  it creates 16x as many patches before tiling. `max_patches=4096` keeps each
  tile bounded for the 24 GB 4090. Lower the multiplier if runtime is too high.
- `center` in `regions_info` is `(x, y)` (column, row) — the OpenCV convention,
  not numpy `(row, col)`; keep this straight when converting to relative point
  prompts for SAM3.
- Merge is deepest-wins: a coarse halo blob overlapping deep detections is
  absorbed (kept only in `n_levels`), so the emitted center/radius always come
  from a real deep region. On the 4K footage the old greedy merge averaged the
  center (74 px off the deep region) and kept the coarse `radius` (139.8 px vs
  the real 42.3 px).
- Cross-level confirmation uses circle overlap OR bbox containment: a wide
  coarse region whose inscribed center sits far from a deep detection still
  confirms it (verified on the 4K footage — circle overlap alone missed the
  largest deep blob). Same-level clustering stays conservative (circle only) so
  two close animals are not fused by one wide box.
- Detections come normalized to the ORIGINAL frame (`letterbox_box` offset
  included), so `det['x'] * frame.size[0]` plots on the raw frame and feeds SAM3
  relative prompts directly; to draw on a manually cropped frame, subtract the
  crop origin from `letterbox_box`.
- `n_levels` counts recursion depths (not views) with a matching region. A
  detection with `n_levels=1` was only seen at one level: `min_confirmations=2`
  drops those, which is a cheap false-positive filter but can drop real objects
  that the coarse view missed.
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
