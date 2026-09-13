# notebooks/01_ingest.ipynb

## Purpose

Dataset ingestion driver. Transcodes every video in `dataset/unprocessed/`
(raw DJI drone files) into `dataset/processed/<recording_timestamp>_<hash>/video.mp4`
(1024x576, 15 fps) with a `metadata.json` sidecar.

## Process

1. `load_env()` + `ROOT = find_repo_root()`.
2. `process_video_dataset(ROOT / "dataset" / "unprocessed", ROOT / "dataset" / "processed")`.

## Inputs / outputs

- Input: `dataset/unprocessed/` (must exist — run `uv run fish-segmentation` first).
- Output: `dataset/processed/<folder>/{video.mp4, metadata.json}` (gitignored).
- Requires system binaries `ffmpeg`/`ffprobe` + `exiftool`.

Details of the functions: [ingest.md](../src/fish_segmentation/ingest.md).
