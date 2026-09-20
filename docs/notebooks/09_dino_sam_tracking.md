# notebooks/09_dino_sam_tracking.ipynb

## Purpose

Bridge DINOv3 candidates to SAM3 point prompts and tracking: run the zoom
detector on one frame, turn each deepest-wins detection into a tracker point
prompt (`obj_id` per detection), propagate the masklets through the video and
export an overlay.

## Process

1. `load_env()` + `ROOT = find_repo_root()`; pick device; same detector settings
   as notebooks 03/07 plus `MIN_CONFIRMATIONS = 2`.
2. Load the video (dataset default, test-media fallback) and the prompt frame
   (frame 0), keeping every frame for visualization. Build a 1024x576 proxy with
   `process_video` for the SAM3 session: SAM3 caches per-frame masks at the
   session resolution and at 4K with several objects that OOMs a 24 GB card,
   while the prompt frame and the overlay keep the full resolution.
3. `run_zoom_detector(..., min_confirmations=2)` → candidate detections with
   normalized `x`/`y`, `bbox`, `n_levels`, `score_mean`, `solidity`; plot the
   candidates and their boxes on the prompt frame.
4. `build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))`
   and `start_session(resource_path=sam3_video_path)`.
5. `add_point_prompt(predictor, session_id, frame_index=0, points=[[x, y]],
   obj_id=k)` once per detection (relative `[0, 1]` coords, one tracked object
   per candidate); visualize the prompted frame.
6. `propagate_in_video(..., start_frame_index=0)` → `save_masklet_video` to
   `notebooks/outputs/dino_prompt_tracking_<dataset>.mp4` (tracker-only sessions
   require the start frame, see `sam3_utils.md`).
7. `close_session` → `predictor.shutdown()`.

## Inputs / outputs

- Input: a video path (dataset default, test media fallback) + gated SAM3/DINOv3
  checkpoints.
- Output: `notebooks/outputs/dino_prompt_tracking_<dataset>.mp4` + the 1024x576
  SAM3 proxy (both gitignored) + plots.

Point prompts cannot be combined with a text prompt in the same `add_prompt`
call, so these masklets track whatever DINO proposed; cross-check against the
`"dolphin"` text-prompt tracks of `04_sam3_track.ipynb`.

Details: [dinov3.md](../src/fish_segmentation/dinov3.md),
[sam3_utils.md](../src/fish_segmentation/sam3_utils.md).
