# notebooks/04_sam3_track.ipynb

## Purpose

The main research pipeline driver: text-prompted video segmentation and dense
tracking with SAM3. Prompts a video with `"dolphin"` on frame 0, propagates
masks across the whole video, and exports a masklet overlay video.

## Process (order matters)

1. `load_env()` + `ROOT = find_repo_root()`; `gpus_to_use = range(torch.cuda.device_count())`.
2. `predictor = build_sam3_video_predictor(gpus_to_use=...)` — first run
   auto-downloads the checkpoint from the gated `facebook/sam3` repo (needs `HF_TOKEN`).
3. `video_frames_for_vis = load_all_frames_rgb(<video>)` (visualization only; the
   model reads the file itself).
4. `start_session(resource_path=...)` → `reset_session` →
   `add_prompt(frame_index=0, text="dolphin")` → visualize first frame →
   `propagate_in_video` → `save_masklet_video(...)` → display →
   `close_session` → `predictor.shutdown()`.

## Inputs / outputs

- Input: `data/test_media/dolphin_00.mp4` (swap for any ingested/enhanced video).
- Output: `notebooks/outputs/output_tracking.mp4` (gitignored).

## Gotchas

- `reset_session` is REQUIRED before switching text prompts in the same session.
- Always `close_session` per video and `predictor.shutdown()` at the end — the
  predictor holds GPU memory otherwise.

Helpers: [sam3_utils.md](../src/fish_segmentation/sam3_utils.md),
frame loading: [video_io.md](../src/fish_segmentation/video_io.md).
