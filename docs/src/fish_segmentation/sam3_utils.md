# fish_segmentation.sam3_utils

## Purpose

Session-level helpers for the SAM3 video predictors (both `build_sam3_video_predictor`
and the multiplex variant — both expose the same `handle_request`/`handle_stream_request` API).

## Functions

| Function | Role |
|---|---|
| `propagate_in_video(predictor, session_id, start_frame_index=None) -> dict[int, outputs]` | Streams `propagate_in_video` requests frame by frame (frame 0 to end) and collects outputs keyed by frame index. Pass `start_frame_index` for tracker-only sessions (see Gotchas). |
| `add_text_prompt(predictor, session_id, frame_index, text) -> outputs` | Wraps an `add_prompt` request with a text prompt; returns the `outputs` dict. |
| `add_point_prompt(predictor, session_id, frame_index, points, labels=None, obj_id=None, rel_coordinates=True) -> outputs` | Wraps an `add_prompt` request with tracker point prompts (one `obj_id` per instance); labels default to all-positive. This is the bridge for `run_zoom_detector` outputs. |

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
