# notebooks/12_dino_sam_chunked.ipynb

## Purpose

Scalable variant of `10_dino_sam_incremental.ipynb`. The single-session
incremental driver grows with objects x frames and OOMs on the full dataset run;
here the proxy is split into `CHUNK_FRAMES` frames (150, 10 s) with
`OVERLAP_FRAMES` of overlap (30, 2 s), **one SAM3 session per chunk**, and every
object alive in the overlap is re-prompted in the next session under the SAME
global `obj_id`, so identities survive the restart. The per-chunk outputs are
merged by global frame index into one continuous masklet video + JSON, so no
manual video concatenation is needed.

## Process

1. `load_env()` + `ROOT = find_repo_root()`; same detector/session knobs as 10
   (`DINO_RESOLUTION_SCALE=4.0`, `KEYFRAME_STRIDE=30`, `MIN_SCORE`/`MIN_SOLIDITY`
   filters, `MAX_OBJECTS` is now a GLOBAL cap, carried objects included) plus the
   chunking knobs `CHUNK_FRAMES`, `OVERLAP_FRAMES`, `CARRY_MIN_AREA`.
2. Build/reuse the 1024x576 proxy, read the original/proxy frame counts, then
   `plan_chunks(n_frames, CHUNK_FRAMES, OVERLAP_FRAMES, align=KEYFRAME_STRIDE)`.
   Each chunk proxy is cut with `process_video(start_time=start/15,
   duration=chunk_len/15)` and its frame count is asserted (`chunk_len`), so
   local `k` maps to global `start + k` exactly. Starts are multiples of 30, so
   the global DINO keyframe grid is the same in every chunk.
3. `load_backbone(...)`; build the SAM3 predictor **once**.
4. Per chunk: `start_session` (with `offload_state_to_cpu=True` /
   `offload_video_to_cpu=True`), then
   - carry over: `carry_over_points(previous_chunk_outputs[start - previous_start],
     min_area=CARRY_MIN_AREA)` gives one interior point per object alive on the
     first frame of the overlap; each is added with
     `add_point_prompt(frame_index=0, obj_id=point["obj_id"])`. If the prompt
     returns an empty mask (`object_mask_area == 0`) the object is dropped; the
     prompt outputs are stored as the frame-0 outputs so `select_new_points`
     rejects detections already covered by the carried masks;
   - incremental loop over local keyframes: DINO on the ORIGINAL at
     `global_k = start + k` → `select_new_points` (quality, containment, bbox
     NMS) → new `obj_id`s from a global counter → `add_prompt` at local `k` →
     `"backward"` (when `k > 0`) then `"forward"`;
   - the chunk propagates at least once (`k=0`, when there are carried objects)
     even if no new detection is accepted, so it always covers its whole range;
   - per-keyframe review figure (cyan = tracked/carried, red = DINO candidates,
     green = prompted, filled yellow = carried) + VRAM print;
   - `close_session` + `torch.cuda.empty_cache()` before the next chunk.
5. `merge_chunk_outputs(chunk_frame_outputs, chunk_plan)` keys everything by
   global frame index, writing chunks oldest to newest: an overlapped frame comes
   from the newer chunk when it produced outputs for it, otherwise the older
   chunk's mask stays (no gaps);
   `save_masklet_video` once on the full proxy, plus `..._tracks.json` (global
   `obj_id` + discovery keyframe/chunk, or `carried_from` for inherited objects)
   and `..._chunks.json` (per-chunk carried/frames/masklets summary).

## Inputs / outputs

- Input: the dataset video (test-media fallback), the 1024x576 proxy, the chunk
  proxies (`notebooks/outputs/chunks/`, gitignored) + gated SAM3/DINOv3
  checkpoints.
- Output: `notebooks/outputs/dino_sam_chunked_<dataset>_<stem>.mp4`,
  `..._tracks.json`, `..._chunks.json` (all gitignored) + keyframe plots.

## Verified end-to-end

- **Smoke, 300 frames of the dataset 4K footage (2 chunk boundaries, plan
  `[(0,150),(120,270),(240,300)]`)**: chunk 0 discovered 12 masklets; chunk 1
  carried 8 of them with the same ids and prompted **0** new ones (the boundary
  dedupe worked); chunk 2 carried 7 (one object had no mask on the hand-over
  frame); 150/150 + 150/150 + 60/60 frames covered, merged 300/300 and the
  exported video has 300 frames.
- **Full dataset run, 475 frames (`CHUNK_FRAMES=150`, `OVERLAP_FRAMES=30`,
  `MAX_OBJECTS=24`)**: 4 chunks, every chunk covered its whole range
  (150/150/150/115), merged **475/475 frames**, **24 masklets** from 98 DINO
  candidates with 20 carried prompts. Per-chunk session VRAM stayed at
  **8.3-9.5 GiB allocated** (peak ~10.5 GiB including DINOv3 + SAM3 weights)
  versus the **18.69 GiB** peak of `10_dino_sam_incremental` on the same video —
  the chunk length is what bounds the objects x frames state.
- The chunked run is not a 1:1 reproduction of 10: each chunk re-runs DINO
  against a fresh session and the chunk proxies add one re-encode, so some
  `select_new_points` containment decisions flip and the total masklet count can
  differ (24 vs 22). The ids are global and stable across chunks; only the count
  and the exact masks change.

## Gotchas

- Memory: `offload_state_to_cpu=True` moves the tracker state to CPU RAM (the
  tensor that grows with objects x frames); `offload_video_to_cpu=True` does NOT
  keep the frame batch off the GPU with the `cv2` loader
  (`_construct_initial_input_batch` moves it back), so the chunk length is what
  bounds the frame + `cached_frame_outputs` side. See
  [sam3_utils.md](../src/fish_segmentation/sam3_utils.md).
- The carry point is the mask's interior point (distance-transform argmax), so
  it is guaranteed to fall inside the animal even for hollow backfilled masks.
  It is computed on the PREVIOUS chunk's first overlap frame; for objects
  discovered inside that overlap the mask there is a backward backfill and can
  be spurious (same caveat as 10, visible in the video).
- Ids are global because each new session reuses the carried `obj_id`; a dolphin
  that vanishes exactly on a chunk boundary and reappears later may still split
  into two ids. Verified case: in the full run, `obj 13` (prompted by chunk 0 at
  global frame 120) got a 4111 px mask there that also suppressed the other
  chunk-0 objects via SAM3's non-overlap constraint, so only 5 of the 14 objects
  had a mask on the hand-over frame; chunk 1 then re-prompted the same detection
  as `obj 14` because the carried mask had shifted off the detection point.
  Chunks 2 and 3 had no such duplicate. Matching tracks post-hoc by mask IoU in
  the overlap (or prompting at the last forward-tracked overlap frame instead of
  the first) is the planned v2 if this shows up more often.
- `MERGE`: chunks are written oldest to newest, so an overlapped frame is taken
  from the newest chunk that actually produced outputs for it (the previous
  chunk's copy survives only if the newer session never reached that frame).
- The `select_new_points` dedupe at `k=0` needs the carried prompt outputs to be
  stored in `outputs_per_frame[0]` — without it the carried dolphins would be
  re-detected and prompted again as new ids.
- Detector settings are tuned for the 4K dataset footage (`resolution_scale=4.0`,
  `min_confirmations=2`). On the native 1024x576 test media (`dolphin_00.mp4`)
  that combination can return zero candidates — notebook 08 uses the default
  `min_confirmations=1` there — so keep the dataset video or lower the filters
  when smoking on test media.
- Same limitations as 09/10: DINO false positives become tracks (no automatic
  filter), masks backfilled before an object appeared can be spurious, and the
  masks come from the 1024x576 proxy, not the 4K original.

Details: [sam3_utils.md](../src/fish_segmentation/sam3_utils.md),
[dinov3.md](../src/fish_segmentation/dinov3.md),
[video_io.md](../src/fish_segmentation/video_io.md).
