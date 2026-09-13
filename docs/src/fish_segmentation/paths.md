# fish_segmentation.paths

## Purpose

Small helpers that remove cwd dependence and centralize `.env` loading. Every
notebook starts with `load_env()` + `ROOT = find_repo_root()`.

## Functions

| Function | Role |
|---|---|
| `find_repo_root() -> Path` | Walks up from the kernel cwd until it finds `pyproject.toml`; raises `FileNotFoundError` if the kernel is outside the repo. |
| `load_env() -> None` | `load_dotenv(<repo root>/.env)` — loads `HF_TOKEN` (and anything else) into the environment before model downloads. |

## Gotchas

- Gated HF repos (`facebook/sam3`, `facebook/sam3.1`, `facebook/dinov3-vitl16-pretrain-lvd1689m`)
  need `HF_TOKEN` present before the first checkpoint download — call `load_env()`
  at the top of notebooks.
- Do not hardcode absolute paths; always compute `ROOT = find_repo_root()`.
