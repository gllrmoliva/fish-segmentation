# notebooks/13_pipeline_e2e.ipynb

## Purpose

Single end-to-end driver of the research flow over the gitignored dataset videos
in `dataset/processed/` (the videos that matter for the project): **INGEST →
optional ENHANCE → DINO → SAM3 → verification**. Unlike 09-12, which are
single-purpose drivers with hardcoded roles, every stage here is toggled from one
parameter cell (`RUN_INGEST`, `RUN_ENHANCE`, `RUN_DINO`, `RUN_SAM`, `SAM_MODE`,
`PROMPT_MODE`, `DINO_SOURCE`) and `START_S`/`DURATION_S` cut a short working clip
for quick runs. All explanatory markdown, plot titles and check messages are in
Spanish (the code identifiers stay English).

## Process

1. **Config cell**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set
   before the first CUDA allocation), `load_env()` + `ROOT = find_repo_root()`,
   the stage toggles, the ingest/enhance knobs, `AnomalyOptions` +
   `DINO_LONG_SIDE`/`DINO_RESOLUTION_SCALE`/`MIN_CONFIRMATIONS`/`KEYFRAME_STRIDE`,
   and the SAM3 knobs (`MAX_OBJECTS`, quality filters, chunk sizes, offload flags).
2. **INGEST**: lists every `dataset/processed/*/video.mp4` with the metadata read
   from the FILE with cv2 (frames, fps, resolution, duration) — not from the
   sidecar, which describes the raw 60 fps original — selects `DATASET_ID` (or
   the newest), optionally runs `process_video_dataset` over
   `dataset/unprocessed/`, cuts the working clip with `process_video`, and
   verifies fps/frames/duration + a 3-frame sample.
3. **ENHANCE (optional)**: `process_marine_video_ffmpeg` (glint inpaint,
   edge-preserving smoothing, CLAHE) into
   `notebooks/outputs/<stem>_enhanced.mp4`, then a before/after/difference plot.
   `DINO_SOURCE` picks what DINO reads (`"processed"` by default, `"enhanced"`
   opt-in).
4. **DINO**: `load_backbone("facebook/dinov3-vitl16-pretrain-lvd1689m")`; the
   `DIAGNOSTIC_KEYFRAME` gets the full diagnostic suite (`plot_detection_results`
   with a coarse `run_dino_view` heatmap/mask, `plot_zoom_overview` of the traced
   zoom tree, `plot_zoom_level` of the root view); then `run_zoom_detector` runs
   over every `KEYFRAME_STRIDE` frame of the working video (indices and the
   1024x576 proxy share the 15 fps grid, so no time remapping is needed) and the
   detections are cached to `<stem>_dino_detections.json`. `RUN_DINO=False`
   reloads that cache instead. Closing plots: candidates per keyframe, score
   histogram, `n_levels` histogram and sample keyframes with the detections.
5. **SAM3**: builds the 1024x576 proxy and the predictor once; `SAM_MODE` selects
   - `chunked` (default): `plan_chunks(n_frames, CHUNK_FRAMES, OVERLAP_FRAMES,
     align=KEYFRAME_STRIDE)`, one session per chunk proxy, `carry_over_points`
     re-prompts every object alive on the first overlap frame with the SAME
     global `obj_id` (drops objects with an empty carried mask),
     `merge_chunk_outputs` by global frame index;
   - `incremental`: one session, per-keyframe `select_new_points` (quality +
     containment + IoU dedupe), `"backward"` then `"forward"` propagation;
   - `simple`: prompt(s) on one frame (`dino_points` or `text`) and a full
     `propagate_in_video`. Text is only supported here: SAM3 cannot mix a text
     and a point prompt in one call;
   - per-keyframe review figure: cyan = tracked masks, red = DINO candidates,
     green = prompted here, filled yellow = carried objects (chunked k=0).
6. **Export & verification**: `save_masklet_video` overlay + `<stem>_tracks.json`
   (per-object origin) + `<stem>_summary.json` (run parameters, per-keyframe log,
   per-chunk summary); per-object visibility/area plots (line of objects per
   frame + `obj_id` x frame area heatmap); a PASS/FAIL checklist per stage with
   counts (keyframes, candidates, tracks, covered frames) and VRAM peak; then
   `close_session` / `predictor.shutdown()` and `nvidia-smi`.

## Inputs / outputs

- Input: a `dataset/processed/<dataset_id>/video.mp4` (must exist; the notebook
  never falls back to `data/test_media/`), its sidecar `metadata.json`, the gated
  SAM3/DINOv3 checkpoints, and optionally raw footage in `dataset/unprocessed/`.
- Output (all gitignored, under `notebooks/outputs/`): the working clip
  (`<stem>_clip_<start>s_<dur>s.mp4`) when sliced, `<stem>_enhanced.mp4`,
  `<stem>_1024x576.mp4` proxy, `<stem>_dino_detections.json`,
  `pipeline_e2e_<stem>_<mode>_<prompt>_tracks.json` / `_summary.json`,
  `pipeline_e2e_<stem>_<mode>_<prompt>.mp4`, and `<proxy_stem>_c<idx>_<start>_<end>.mp4`
  chunk proxies for the chunked mode.

## Verified end-to-end

- **Smoke, 20 s (300 frames) of `20260813_213529_7f994a2f` (4K, 15 fps),
  `RUN_ENHANCE=True` (DINO read the processed source), `RUN_DINO=True`,
  `RUN_SAM=True`, `SAM_MODE="chunked"`**: nbconvert finished with **0 errors** and
  all **8 checklist entries OK**. DINO ran on 10 keyframes → **45 candidates**
  (4/3/4/5/4/4/3/8/4/6). Chunk plan `[(0,150),(120,270),(240,300)]`: chunk 0
  discovered **9 masklets** (from its 20 candidate evaluations), chunk 1 carried
  9 ids and prompted 3 new (→12), chunk 2 carried 9 and prompted 2 new (→14).
  32 track-log records (14 discovered + 18 carried), `merge_chunk_outputs`
  covered **300/300 frames**, mean **9.88 visible objects/frame**, and 100 % of
  frames had at least one mask. Per-chunk session memory stayed at **7.0-8.8 GiB
  allocated** (9.62 GiB peak for the whole run, DINOv3 + SAM3 + 3 sessions on a
  24 GB 4090). Artifacts written: `pipeline_e2e_<stem>_chunked_dino_points.mp4`,
  `..._tracks.json`, `..._summary.json`, the DINO detections cache, the 1024x576
  proxy and 21 rendered figures across the sections.
- The committed notebook differs from that exact executed copy only in the
  defensive guard of the DINO summary cell (`any(len(...))` over the cached
  detections); the empty-cache branch was checked separately, the validated path
  is identical.

## Gotchas

- **ENHANCE is expensive at 4K**: `process_marine_video_ffmpeg` ran at
  ~2.5 s/frame (≈13 min for this 300-frame clip). Keep `RUN_ENHANCE=False`
  unless `DINO_SOURCE="enhanced"` is actually being tested.
- The frame indices of the DINO source and the SAM3 proxy are used 1:1: both are
  cut from the same 15 fps working clip, so keyframe `k` exists in both. If
  `RUN_ENHANCE=True` the enhanced file is a re-encode and its frame count is
  asserted equal by construction (same duration/fps), but the proxy is built
  from the *processed* working clip on purpose (natural colors for the overlay).
- `RUN_DINO=False` requires a previous `<stem>_dino_detections.json`; otherwise
  SAM3 gets zero candidates and only the text/simple mode produces tracks.
- `PROMPT_MODE="text"` asserts `SAM_MODE="simple"` in the incremental/chunked
  cells; `reset_session` would be needed to switch prompts mid-session.
- `CHUNK_FRAMES - OVERLAP_FRAMES` must stay a multiple of `KEYFRAME_STRIDE` so
  the global keyframe grid does not shift between chunks (`plan_chunks` raises).
- Same limitations as 09-12: DINO false positives become tracks, backfilled
  masks before an object appeared can be spurious, and the mask borders come from
  the 1024x576 proxy.

Details: [dinov3.md](../src/fish_segmentation/dinov3.md),
[sam3_utils.md](../src/fish_segmentation/sam3_utils.md),
[ingest.md](../src/fish_segmentation/ingest.md),
[enhance.md](../src/fish_segmentation/enhance.md).
