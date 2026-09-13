# fish_segmentation.sam3_utils

## Purpose

Session-level helpers for the SAM3 video predictors (both `build_sam3_video_predictor`
and the multiplex variant — both expose the same `handle_request`/`handle_stream_request` API).

## Functions

| Function | Role |
|---|---|
| `propagate_in_video(predictor, session_id) -> dict[int, outputs]` | Streams `propagate_in_video` requests frame by frame (frame 0 to end) and collects outputs keyed by frame index. |
| `add_text_prompt(predictor, session_id, frame_index, text) -> outputs` | Wraps an `add_prompt` request with a text prompt; returns the `outputs` dict. |

## Gotchas

- Session ordering: `start_session` → `add_prompt` → `propagate_in_video` →
  `close_session` → `predictor.shutdown()`.
- `reset_session` (raw `handle_request` call) is REQUIRED before switching to a
  different text prompt in the same session, or results are silently wrong.
