# notebooks/10_dino_sam_incremental.ipynb

## Purpose

Incremental variant of `09_dino_sam_tracking.ipynb`: instead of computing the DINO
candidates once at frame 0, the zoom detector runs on every `KEYFRAME_STRIDE`
frames (30 by default, 2 s at 15 fps) over the 1024x576 SAM3 proxy and every
candidate not already covered by a tracked mask is added to the **live** SAM3
session as a new `obj_id`. New objects are backfilled to the start of the video
and then tracked forward, so dolphins that appear mid-video get their own
masklets without recomputing the existing tracks.

## Process

1. `load_env()` + `ROOT = find_repo_root()`; device; same detector settings as
   09 plus the incremental knobs: `KEYFRAME_STRIDE`, `MAX_OBJECTS`,
   `MIN_SCORE`/`MIN_SOLIDITY`/`MIN_AREA`, `MASK_MARGIN_PX`, `DEDUPE_OVERLAP`.
2. Build/reuse the 1024x576 proxy with `process_video` for the SAM3 session
   (4K per-frame mask caches OOM a 24 GB card) and read the original/proxy
   frame counts + original fps with OpenCV. Keyframe k of the session (15 fps)
   maps to the original by time: `orig_index = k * orig_fps / SESSION_FPS`.
3. `load_backbone("facebook/dinov3-vitl16-pretrain-lvd1689m")`; build the SAM3
   predictor and `start_session(resource_path=proxy)`.
4. Incremental loop over `keyframes = range(0, n_frames, KEYFRAME_STRIDE)`:
   `load_frame_rgb` on the ORIGINAL at `orig_index` (the 1024x576 proxy yields
   **zero** DINO detections on the dataset footage — verified — because the
   downscale destroys the small-object signal) → `run_zoom_detector` →
   `select_new_points` (quality filters, containment in the masks tracked on
   that frame, bbox-IoU NMS against the other survivors) → one
   `add_point_prompt` per survivor with
   `obj_id = next_obj_id(outputs_per_frame) + i` → propagate. From keyframe 1 on
   the order is `"backward"` (backfill 0..k-1) and then `"forward"` (k..end);
   both passes are needed because a single `"both"` call does not backfill newly
   added objects (SAM3 action history).
5. Per-keyframe review figure (keyframe downscaled to 1024x576): tracked masks
   as cyan contours, all DINO candidates in red, prompted ones in green.
6. `save_masklet_video` on the proxy frames + JSON track log (discovery
   keyframe and detection metadata per `obj_id`); `close_session` →
   `predictor.shutdown()`.

## Inputs / outputs

- Input: a video path (dataset default, test-media fallback) + gated SAM3/DINOv3
  checkpoints.
- Output: `notebooks/outputs/dino_sam_incremental_<dataset>_<stem>.mp4`,
  `..._tracks.json` and the 1024x576 proxy (all gitignored) + keyframe plots.

## Verified end-to-end (smoke)

On an 8 s segment of the dataset 4K footage: 5 DINO detections at frame 0, then
1/2/2 new objects prompted at keyframes 30/60/90 (the rest deduped against the
tracked masks); 120/120 frames covered; the mid-video objects got masks before
their discovery frame (backfill 21/30, 60/60 and 90/90 frames respectively) and
after it; the existing tracks at frame 10 were unchanged (0 pixels).

## Gotchas

- DINO on the 1024x576 proxy detects nothing on this footage at any
  `resolution_scale` (1.0/2.0/4.0) — the keyframes must come from the original.
  Original and proxy are both aspect-preserving, so the relative detections feed
  the session unchanged.
- Partial propagation is per new object, but the masks of already tracked
  objects are fetched from `cached_frame_outputs` — they are never recomputed.
- The session grows with objects x frames (~0.3 GB of GPU cache + ~0.3 GB of
  Python-side masks per object over 475 frames at 1024x576); `MAX_OBJECTS` caps
  it and extra candidates are dropped for that keyframe.
- Backfilled masks on frames before an object actually appeared can be spurious;
  they cannot be filtered by score because SAM3 does not expose a per-frame
  tracker score in the outputs (empty masks are dropped).
- `MIN_AREA` defaults to `None`: `area` is in view pixels and is not comparable
  across zoom levels. `score_mean` and `solidity` are.
- DINO false positives (glints, waves) still become tracks — the same limitation
  as 09; cross-check against the `"dolphin"` text-prompt tracks of
  `04_sam3_track.ipynb`.

Details: [dinov3.md](../src/fish_segmentation/dinov3.md),
[sam3_utils.md](../src/fish_segmentation/sam3_utils.md),
[video_io.md](../src/fish_segmentation/video_io.md).
