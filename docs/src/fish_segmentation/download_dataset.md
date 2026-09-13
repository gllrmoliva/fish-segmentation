# fish_segmentation.download_dataset

## Purpose

Fetches the raw dataset (DJI drone dolphin footage) from a Google Drive folder
into `dataset/` using `gdown`. Creates `dataset/` if missing. This is the
prerequisite step for [ingest.md](ingest.md), which reads `dataset/unprocessed/`.

Exposes `main()`, wired to the `fish-segmentation` console script.

## Contents

- `main()` — the whole process:
  - `folder_id = "1gWVwK3860mkuir6n-GpxoeOXUtKIXy9i"` (hardcoded Drive folder).
  - target: `<repo root>/dataset` (resolved via `fish_segmentation.paths.find_repo_root()`).
  - `gdown.download_folder(url, output=..., quiet=False, use_cookies=False)`.

## Usage

```bash
uv run fish-segmentation          # console script
uv run python -m fish_segmentation.download_dataset   # equivalent
```

## Gotchas

- Starts a real, large download — don't run it casually.
- Large, unresumable folder download; if gdown hits a Drive rate limit it may
  partially fail — re-running re-downloads (no local skip logic).
- Google Drive folder contents mirror directly, so the script has no knowledge of
  the `unprocessed/`/`processed/` split: files land wherever the Drive folder puts
  them; ingestion expects them under `dataset/unprocessed/`.
- `dataset/` is gitignored — never commit its contents.
