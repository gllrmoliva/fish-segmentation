# notebooks/11_dino_erase_rerun.ipynb

## Purpose

Single-frame experiment: try to surface *weaker* DINO anomalies (e.g. submerged
dolphins) that lose against the dominant heatmap peak. Four variants over the
same 4K frame: the untouched baseline, erase + re-forward (`inpaint` and `patch`)
and in-model suppression (`score` and `tokens`). An A/B/analysis driver, not part
of the ingest → detect → track flow.

Why it can work: `compute_anomaly_heatmap` builds its reference pool from the
frame's own tokens (a strong object contaminates the "water" reference and
flattens the contrast of the rest) and the mask thresholds are relative
percentiles (99.5 seed / 98 growth), so the second peak only becomes detectable
once the first is removed. Honest limit: the loop does **not** invent signal, and
`run_zoom_detector` already re-runs DINO on nested zooms — hence the baseline.

## Process

1. `load_env()` + `ROOT = find_repo_root()`; config: `FRAME_INDEX`, detector
   knobs (`DINO_LONG_SIDE=1024`, `DINO_RESOLUTION_SCALE=4.0`,
   `ZONE_MERGE_GAP=0.08`, `MIN_ZONE_FRACTION=0.25`, `MIN_CONFIRMATIONS=2`) and
   loop knobs (`MAX_ITERATIONS=2`, `ERASE_DILATE_PX=12`, `MIN_SCORE=0.5`,
   `MIN_SOLIDITY=0.25`).
2. `load_frame_rgb` on the ORIGINAL video (dataset processed, test-media
   fallback) + `load_backbone`.
3. SAM3 image model (`build_sam3_image_model` + `Sam3Processor`) and
   `mask_fn = partial(detection_erase_mask, processor=sam3_processor,
   max_area_fraction=0.25, min_score=0.5)`.
4. `run_dino_erase_loop(erase_mode="inpaint")`,
   `run_dino_erase_loop(erase_mode="patch")`,
   `run_dino_suppress_loop(suppress_mode="score")` and
   `run_dino_suppress_loop(suppress_mode="tokens")`, each followed by
   `plot_erase_records` (previous erasure red, new erasure yellow, accepted
   candidates lime, leaked candidates red ✕).
5. Summary JSON (detection metadata only) + comparison table (`new` / `leaked` /
   mean score per iteration) and one crop per iteration around the erased region
   to check the inpaint hole / patch seam by eye.
6. Cleanup: `del` of the models + `torch.cuda.empty_cache()`.

## Inputs / outputs

- Input: `dataset/processed/20260813_213529_7f994a2f/video.mp4` (fallback
  `data/test_media/dolphin_00.mp4`) + gated DINOv3/SAM3 checkpoints.
- Output: `notebooks/outputs/dino_erase_rerun_<dataset>_<stem>_frame<N>.json`
  (gitignored) + the inline figures.

## Verified end-to-end (smoke)

Frame 0 of `dataset/processed/20260813_213529_7f994a2f/video.mp4` (3840x2160,
RTX 4090, peak VRAM 7.49 GB): baseline 5 detections (mean `score_mean` 0.742).
With `MAX_ITERATIONS=2` and `ERASE_DILATE_PX=12`, every mode found 2-3
additional candidates outside the erased/suppressed masks and **zero leaked**
detections:

| mode | it1 new | it1 mean score |
|---|---|---|
| erase_inpaint | 2 | 0.590 |
| erase_patch | 3 | 0.739 |
| suppress_score | 2 | 0.705 |
| suppress_tokens | 2 | 0.765 |

What the new candidates are matters more than the count: the erase modes add a
compact one at (0.529, 0.470) — the crops in the notebook show it in the same
pod area as the baseline detections — while all four modes also pick up a large
(~16k px) elongated (solidity ~2.7-3.3) region at (0.485, 0.60), most likely the
residual of the erased group rather than a new animal. The suppression modes
never touch pixels and keep the highest-scoring candidates, so they are the
cheapest first try; `leaked_detections` stayed at 0 for every mode.

## Gotchas

- `leaked_detections` is the key metric: a candidate inside an already-erased
  region is an erase artifact (inpaint hole, pasted-texture seam), not a new
  animal. It is the signal that an erase mode is not working.
- The erase mask is built from SAM3 box prompts with a DINO bbox/circle fallback
  (`detection_erase_mask`), and is dilated by `ERASE_DILATE_PX` (≥ 1 ViT patch at
  the detector's scale) so no mixed patch is left at the object border.
- The suppression modes never touch pixels: `"tokens"` replaces the found patch
  tokens with the re-normalized mean of the survivors (zeroing them would score
  as *maximum* anomaly).
- The notebook enters `torch.autocast("cuda", dtype=torch.bfloat16)` in cell 1
  (same as `05_smoke_test.ipynb`): the vendored SAM3 image-model ViT MLP
  hardcodes bf16 activations, so the image model cannot run in plain fp32.
- The summary JSON stores detection metadata only; the figures are not persisted
  by the notebook (they are embedded in the `.ipynb` on execution).
- The notebook runs four full zoom-detector loops (plus SAM3 image inference) on
  one 4K frame — expect several minutes on the 4090, mostly detector runtime.

Details: [dinov3.md](../src/fish_segmentation/dinov3.md),
[sam3_utils.md](../src/fish_segmentation/sam3_utils.md),
[video_io.md](../src/fish_segmentation/video_io.md).
