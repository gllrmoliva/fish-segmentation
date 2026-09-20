# AGENTS.md

Research project: segment and track dolphins in aerial drone footage using Meta
SAM3 (text-prompted video segmentation/tracking) and DINOv3 (anomaly-based object
detection). All functions/classes live in the `fish_segmentation` package
(`src/fish_segmentation/`); notebooks in `notebooks/` are thin drivers (no
function definitions). Tests live in `tests/` (pytest-style; `test_zoom_detector.py`
also runs standalone). No lint config, no CI.

Per-file explanations live in `docs/` — start at `docs/index.md`.

## Environment

- uv workspace: `sam3_src/` is a vendored clone of facebookresearch/sam3 mounted as
  an editable workspace member providing the `sam3` package. It is library code,
  NOT project code — do not document or refactor it. Note: the vendored copy has
  local patches (e.g. `sam3/perflib/fa3.py` switched fp8 → bf16, "FIX para flash attention"),
  so it is not upstream-identical.
- Python pinned `==3.12.*` (`.python-version`); environments restricted to Linux.
  Install: `uv sync` (large download: torch 2.10 + torchvision + flash-attn-3
  from the PyTorch cu128 index).
- System deps: `ffmpeg` (incl. `ffprobe`) and `exiftool`.
- Copy `.env.example` → `.env` and set `HF_TOKEN`: SAM3 checkpoints auto-download
  from gated HF repos (`facebook/sam3`, `facebook/sam3.1`) on first model build;
  DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is also gated. Call
  `load_env()` (from `fish_segmentation.paths`) at the top of every notebook.
- Run notebooks: `uv run jupyter lab` from the repo root. Kernel (already named in README):
  `uv run python -m ipykernel install --user --name=fish-segmentation --display-name="smell-like-fish"`.

## Target hardware

- Runs on a single RTX 4090 (24 GB). No multi-GPU code paths assumed; code uses
  `range(torch.cuda.device_count())`, which resolves to 1 GPU there.
- Keep VRAM-conscious defaults when adding code: `offload_state_to_cpu=True` on
  `start_session`, `compile=False`, `warm_up=False`, and ingest/downscale video
  before inference (ingestion normalizes to 1024x576 @ 15 fps).
- If flash-attn-3 (`use_fa3=True`, the default for video predictors) fails to import
  or launch on Ada/sm_89, rebuild with `use_fa3=False` — there is a verified SDPA
  fallback (`sam3/model/model_misc.py:397`).

## Repo map

- `src/fish_segmentation/` — the package (editable install; new modules are
  picked up without re-sync):
  - `paths.py` — `find_repo_root()` (walks up to `pyproject.toml`; removes any
    cwd dependence) + `load_env()` (`load_dotenv(<root>/.env)`).
  - `video_io.py` — `load_all_frames_rgb()` (all frames, numpy/PIL-paths for SAM3
    visualization) and `load_sampled_frames()` (n equally spaced PIL frames).
  - `sam3_utils.py` — `propagate_in_video()`, `add_text_prompt()`,
    `add_point_prompt()`.
  - `ingest.py` — DJI dataset ingestion (transcode to 1024x576 @ 15 fps + exiftool/
    ffprobe metadata sidecar).
  - `enhance.py` — marine video enhancement (deglint/inpaint, edge-preserving
    smoothing, CLAHE).
  - `dinov3.py` — DINOv3 anomaly detection pipeline (heatmap → mask → salient
    regions with guaranteed-interior points, coarse-to-fine zoom with
    deepest-wins merge).
  - `download_dataset.py` — `main()` behind the `fish-segmentation` console
    script: fetch raw dataset from a hardcoded Google Drive folder into `dataset/`.
- `notebooks/` — driver notebooks (process only): `01_ingest`, `02_enhance`,
  `03_dinov3_detect`, `04_sam3_track` (main SAM3 pipeline), `05_smoke_test`
  (SAM3 constructor-argument reference), `07_dinov3_zoom_levels` (zoom trace),
  `08_dinov3_heatmap_ab` (score/mask A/B), `09_dino_sam_tracking` (DINO points
  as SAM3 tracker prompts), `10_dino_sam_incremental` (DINO every N frames
  against the live SAM3 session: new candidates become new masklets mid-video).
  Outputs go to `notebooks/outputs/` (gitignored).
  `notebooks/reference/sam3_video_predictor.ipynb` is the upstream Meta demo,
  reference only — do not execute or develop in it.
- `tests/` — CPU-only pytest-suite (heatmap, zoom geometry/merge, SAM3 request
  wrappers); `tests/test_zoom_detector.py` also runs as a standalone script.
- `data/test_media/` — git-tracked sample media (dolphin_00..03.mp4, dogs.jpg,
  dogs_playing.mp4). Default input for the driver notebooks.
- `dataset/` — gitignored, local only. `unprocessed/` = raw DJI footage;
  `processed/` = normalized output of ingestion. Do not confuse with `data/`.

## Gotchas

- `uv run fish-segmentation` runs the Google Drive dataset download — it starts
  a real, large download; don't run it casually.
- SAM3 video session API ordering matters: `start_session` → `add_prompt` →
  `propagate_in_video` → `close_session` → `predictor.shutdown()`. You MUST
  `reset_session` before switching to a different text prompt in the same session,
  or results are wrong.
- SAM3 point prompts cannot be mixed with a text/box prompt in one `add_prompt`
  call (`add_point_prompt()`), and a tracker-only session must pass
  `start_frame_index` to `propagate_in_video` — SAM3 otherwise raises
  "No prompts are received on any frames".
- `run_zoom_detector` detections are normalized to the ORIGINAL frame (letterbox
  bars included) with `x`/`y`, `bbox`, `level`, `n_levels` and region metadata, so
  they feed SAM3 point prompts directly (`rel_coordinates=True`).
- `find_repo_root()` depends on the kernel cwd being inside the repo — don't
  hardcode absolute paths in notebooks; compute from `ROOT`.
- `dataset/` and `notebooks/outputs/` are gitignored — never commit media or
  derived outputs.
- Commit messages in history are terse ("." / short phrases); no branch/PR conventions.
