# fish_segmentation.sam3_utils

## Purpose

Session-level helpers for the SAM3 video predictors (both `build_sam3_video_predictor`
and the multiplex variant — both expose the same `handle_request`/`handle_stream_request` API),
plus the image-model box helpers used to build DINO erase masks.

## Functions

| Function | Role |
|---|---|
| `propagate_in_video(predictor, session_id, start_frame_index=None, propagation_direction="both", max_frame_num_to_track=None) -> dict[int, outputs]` | Streams `propagate_in_video` requests frame by frame (frame 0 to end) and collects outputs keyed by frame index. Pass `start_frame_index` for tracker-only sessions (see Gotchas) and an explicit direction in the incremental loop. |
| `add_text_prompt(predictor, session_id, frame_index, text) -> outputs` | Wraps an `add_prompt` request with a text prompt; returns the `outputs` dict. |
| `add_point_prompt(predictor, session_id, frame_index, points, labels=None, obj_id=None, rel_coordinates=True) -> outputs` | Wraps an `add_prompt` request with tracker point prompts (one `obj_id` per instance); labels default to all-positive. This is the bridge for `run_zoom_detector` outputs. |
| `select_new_points(detections, frame_outputs=None, min_score=None, min_solidity=None, min_area=None, mask_margin_px=0, dedupe_overlap=0.3) -> list[dict]` | Filters DINO detections down to points that should start new masklets: quality thresholds, containment in the masks tracked on that frame, and bbox-IoU NMS among the survivors (highest `score_mean` wins). Used by the incremental notebook. |
| `point_in_existing_mask(det, frame_outputs, mask_margin_px=0) -> bool` | True when a detection center falls inside a tracked mask of that frame; `mask_margin_px` widens every mask (square window) so a point on a tracked animal's edge still counts as covered. Off-frame points are treated as covered. |
| `next_obj_id(outputs_per_frame) -> int` | Smallest unused `obj_id` across a frame→outputs mapping (`0` when empty). |
| `segment_boxes(processor, image, boxes, label=True) -> list[dict]` | Runs the SAM3 image-model processor (`Sam3Processor`) with one normalized `[cx, cy, w, h]` box prompt at a time — `reset_all_prompts` between boxes — and keeps the best mask per box: `{'mask': bool HxW or None, 'score': float}`. Duck-typed (no sam3 import). |
| `detection_erase_mask(detections, image, processor, max_area_fraction=0.25, min_score=0.5, fallback_shape="bbox", fallback_radius_scale=1.5) -> bool HxW` | Erase mask for DINO detections: one SAM3 box prompt per detection (converted with `_bbox_to_cxcywh`), falling back to the DINO bbox/circle when SAM3 returns no mask, a score below `min_score` or an area above `max_area_fraction` of the frame (a class-agnostic box on open water can segment the whole crop). The signature matches `run_dino_erase_loop`'s `mask_fn(detections, image)` convention, so `partial(detection_erase_mask, processor=...)` plugs in directly. |

## Gotchas

- Session ordering: `start_session` → `add_prompt` → `propagate_in_video` →
  `close_session` → `predictor.shutdown()`.
- `reset_session` (raw `handle_request` call) is REQUIRED before switching to a
  different text prompt in the same session, or results are silently wrong.
- Point prompts and text/box prompts cannot be mixed in one `add_prompt` call
  (`sam3/model/sam3_video_inference.py:1374` asserts `text_str is None and
  boxes_xywh is None` when points are given). Call again with the same `obj_id`
  to refine an object, or with a new `obj_id` to add another one; the predictor
  tracks at most `max_num_objects=128` masklets per session.
- With `rel_coordinates=True` (SAM3 default) points are relative `[0, 1]` of the
  ORIGINAL frame — exactly the convention of `run_zoom_detector` detections, so
  `points=[[det['x'], det['y']]]` needs no rescaling.
- A tracker-only session (point prompts, no text/box prompt) must pass
  `start_frame_index` to `propagate_in_video` — the frame where the points were
  added. Without it the vendored SAM3 processing-order selection raises
  `No prompts are received on any frames` because its detector-stage prompt list
  is empty (verified end-to-end).
- Each point prompt immediately runs tracker inference on that frame and returns
  its masks; unsupported/invalid prompts fail loudly rather than silently.
- Adding points AFTER a propagation works (the built model is
  `Sam3VideoInferenceWithInstanceInteractivity`): `add_prompt(points, obj_id)`
  goes through `add_tracker_new_points` and the next propagation becomes a
  *partial* tracker propagation for the new obj_ids, merged with
  `cached_frame_outputs` — existing tracks are not recomputed.
- Direction matters when adding mid-video: with `propagation_direction="both"`,
  the forward pass is `propagation_partial` but the reverse pass degenerates into
  `propagation_fetch` (cached outputs only), so it does NOT backfill the new
  object. Call `"backward"` (fills 0..k-1) and then `"forward"` (fills k..end)
  explicitly. Verified in `parse_action_history_for_propagation`
  (`sam3_video_inference.py:1233`).
- `out_binary_masks` are boolean masks at session resolution; masks with zero
  area are dropped from the outputs (`_postprocess_output`), and there is no
  per-frame tracker score exposed, so backfilled garbage masks cannot be filtered
  by score downstream.
- `segment_boxes`/`detection_erase_mask` are the IMAGE-model path, not the video
  session helpers: they take a `Sam3Processor` (see `notebooks/05_smoke_test.ipynb`)
  and a PIL image, and never touch a session.
- `segment_boxes` resets the geometric prompts after every box because
  `Sam3Processor` accumulates them in the same state — without the reset the
  second box would segment the union of both.
- `detection_erase_mask` skips SAM3 for detections without `bbox` and goes
  straight to the fallback (bbox or circle), so it also works for region-style
  detections; the accepted area cap is checked against the whole frame.
- SAM3 image-model inference must run under `torch.autocast("cuda",
  dtype=torch.bfloat16)` (as `notebooks/05_smoke_test.ipynb` and
  `notebooks/11_dino_erase_rerun.ipynb` do): the vendored ViT MLP
  (`sam3/perflib/fused.py:addmm_act`) hardcodes bf16 activations against fp32
  weights, so without autocast the first `fc2` raises a dtype mismatch, and the
  bf16 outputs then need the `_to_numpy` fp32 cast before `.numpy()`.
