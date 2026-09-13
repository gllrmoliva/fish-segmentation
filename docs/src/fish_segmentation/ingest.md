# fish_segmentation.ingest

## Purpose

Dataset ingestion. Walks `dataset/unprocessed/` (raw DJI drone files), reads real
recording metadata, and normalizes every video into the layout used by the rest
of the project:

```
dataset/processed/<recording_timestamp>_<8-hex-path-hash>/
  video.mp4        # 1024x576, 15 fps, libx264, yuv420p (letterboxed to keep aspect ratio)
  metadata.json    # exiftool + ffprobe dump + recording_timestamp
```

Driver: `notebooks/01_ingest.ipynb`. Input videos must exist first — run
`uv run fish-segmentation` (see [download_dataset.md](download_dataset.md)).

## Functions

| Function | Role |
|---|---|
| `process_video(input_path, output_path, target_width, target_height, start_time=None, duration=None, fps=None)` | ffmpeg-python transcode: scale to fit canvas + pad black bars, optional `fps`/clip. Output always libx264 + aac + yuv420p. |
| `extract_datetime_from_filename(filename) -> str \| None` | Fallback timestamp from the DJI naming pattern `DJI_YYYYMMDDHHMMSS_...`. |
| `get_recording_datetime_and_metadata(file_path) -> (str, dict)` | Builds the full metadata dict (`file_system`, `exif_metadata`, `ffprobe_streams`) and resolves the true recording time. Timestamp priority: QuickTime/EXIF/XMP date tags → filename regex → filesystem mtime. |
| `generate_unique_folder_name(file_path, recording_timestamp) -> str` | Output folder name: `YYYYMMDD_HHMMSS` + 8-char sha256 of the resolved source path (deduplicates same-timestamp recordings). |
| `process_video_dataset(input_dir, output_base_dir)` | Recurses `rglob("*")`, keeps video extensions (`.mp4 .mov .avi .mkv .webm .ts`), and for each file writes `metadata.json` before transcoding. |

## Gotchas

- Needs the system binaries `ffmpeg`/`ffprobe` and `exiftool` (`pyexiftool` shells
  out to it); both fail softly — errors are stored inside `metadata.json` instead
  of aborting the run.
- Scale + pad means non-16:9 inputs get black bars baked in rather than cropped.
- Re-running overwrites (`overwrite_output=True`) and reuses the same folder name
  (hash of source path), so processing is idempotent per input file.
- The `recording_timestamp` in `metadata.json` is the folder-name authority; other
  fields are diagnostics.
