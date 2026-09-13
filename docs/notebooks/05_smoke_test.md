# notebooks/05_smoke_test.ipynb

## Purpose

Smoke tests / constructor-argument reference for the two SAM3 entry points,
using the bundled `data/test_media/` dogs media (not dolphin footage). Copy
parameter choices from here into real pipelines.

## Process

1. Device/precision setup: bf16 autocast on CUDA + TF32 for compute capability
   ≥ 8 (the 4090 qualifies); `load_env()` + `ROOT = find_repo_root()`.
2. **Image path**: `build_sam3_image_model()` + `Sam3Processor` →
   `set_image(dogs.jpg)` / `set_text_prompt("dogs")` → masks/boxes/scores
   visualization.
3. **Video path**: `build_sam3_multiplex_video_predictor(checkpoint_path=None,
   bpe_path=None, max_num_objects=16, multiplex_count=16, use_fa3=True,
   use_rope_real=True, compile=False, warm_up=False, session_expiration_sec=1200,
   default_output_prob_thresh=0.5, async_loading_frames=True)` →
   `start_session(..., offload_state_to_cpu=True)` → `reset_session` →
   `add_text_prompt("dog")` → `propagate_in_video` → visualize every 60 frames →
   `close_session` → `predictor.shutdown()`.

## Gotchas

- `use_fa3=True` pulls flash-attn-3 (`sam3/perflib/fa3.py`); if it fails on the
  4090 (Ada/sm_89), rebuild with `use_fa3=False` (SDPA fallback).
- This notebook documents the exact multiplex-predictor signature in use — keep
  it in sync when upgrading `sam3`.
