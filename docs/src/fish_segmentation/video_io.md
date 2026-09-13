# fish_segmentation.video_io

## Purpose

Frame loading for both pipelines. Accepts an `.mp4` path or a folder of
`<frame_index>.jpg` files (numerically sorted, with lexicographic fallback).

## Functions

| Function | Role |
|---|---|
| `load_all_frames_rgb(video_path) -> list[np.ndarray] \| list[str]` | All frames as RGB numpy arrays (MP4 input) or sorted JPEG paths (frame-folder input). For SAM3 visualization only — the model reads the video file itself via `resource_path`. |
| `load_sampled_frames(video_path, n_frames) -> list[PIL.Image]` | `n_frames` equally spaced frames as PIL Images (DINOv3 pipeline input). Empty list if the video reports no frames. |

## Gotchas

- The two functions have different return types by design (numpy for OpenCV-style
  visualization, PIL for the transformers processor) — don't "unify" them blindly.
- JPEG-folder sorting falls back to lexicographic order and prints a warning if
  names are not `<frame_index>.jpg`.
