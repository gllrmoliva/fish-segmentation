# docs/

Per-file explanations for the project code. `sam3_src/` is a vendored copy of
facebookresearch/sam3 used as a library — it is intentionally not documented here.

## Reading order

The research flow is: **ingest → enhance → detect → segment/track**.

1. [Get data](notebooks/01_ingest.md) — `uv run fish-segmentation` (or
   `src/fish_segmentation/download_dataset.py`) pulls raw DJI drone videos from
   Google Drive into `dataset/unprocessed/`.
2. [Ingest](notebooks/01_ingest.md) — `notebooks/01_ingest.ipynb` transcodes raw
   footage into `dataset/processed/<timestamp>_<hash>/video.mp4` (1024x576 @ 15 fps)
   with a `metadata.json` sidecar.
3. [Enhance](notebooks/02_enhance.md) — `notebooks/02_enhance.ipynb` cleans marine
   footage (glint inpainting, edge-preserving smoothing, CLAHE).
4. [Detect](notebooks/03_dinov3_detect.md) — `notebooks/03_dinov3_detect.ipynb`
   uses DINOv3 patch-token anomaly detection to find salient regions and interior
   points (candidate point prompts).
5. [Segment/track](notebooks/04_sam3_track.md) — `notebooks/04_sam3_track.ipynb`
   is the main SAM3 video pipeline: text prompt "dolphin" → dense tracking masks
   → masklet video.

Support files:

- [notebooks/05_smoke_test.md](notebooks/05_smoke_test.md) — SAM3 image model +
  multiplex video predictor constructor-argument reference.
- [notebooks/07_dinov3_zoom_levels.md](notebooks/07_dinov3_zoom_levels.md) —
  trace-driven layer-by-layer view of the DINO zoom detector.
- [notebooks/08_dinov3_heatmap_ab.md](notebooks/08_dinov3_heatmap_ab.md) —
  score-mode/local-contrast and mask-method A/B for the DINO detector.
- [notebooks/09_dino_sam_tracking.md](notebooks/09_dino_sam_tracking.md) — DINO
  candidate points as SAM3 point prompts + masklet tracking.
- [notebooks/10_dino_sam_incremental.md](notebooks/10_dino_sam_incremental.md) —
  DINO every N frames against the live SAM3 session: new candidates are added as
  new masklets mid-video (backfill + forward) instead of prompting only once.
- `notebooks/reference/sam3_video_predictor.ipynb` — upstream Meta demo notebook,
  reference only.

Package modules (all functions/classes live here):

- [paths.md](src/fish_segmentation/paths.md) — repo-root discovery + `.env` loading
- [video_io.md](src/fish_segmentation/video_io.md) — frame loading for both pipelines
- [sam3_utils.md](src/fish_segmentation/sam3_utils.md) — SAM3 session helpers
- [ingest.md](src/fish_segmentation/ingest.md) — dataset normalization
- [enhance.md](src/fish_segmentation/enhance.md) — marine video enhancement
- [dinov3.md](src/fish_segmentation/dinov3.md) — anomaly detection pipeline
- [download_dataset.md](src/fish_segmentation/download_dataset.md) — raw dataset fetch / CLI

## Conventions

- Driver notebooks contain process only (no function/class definitions); call
  `load_env()` and compute `ROOT = find_repo_root()` in the first cell of every notebook.
- `.env` must hold `HF_TOKEN` (gated SAM3/DINOv3 checkpoints).
- Target machine is a single RTX 4090 (24 GB) — keep VRAM-conscious settings
  (`offload_state_to_cpu=True`, `compile=False`, `warm_up=False`, downscaled input).
- Derived artifacts go to `notebooks/outputs/` (gitignored); `data/` holds
  git-tracked sample media; `dataset/` is the gitignored research dataset.
