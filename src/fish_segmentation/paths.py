from pathlib import Path

from dotenv import load_dotenv


def find_repo_root() -> Path:
    for parent in (Path.cwd(), *Path.cwd().parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("no pyproject.toml found above the current working directory")


def load_env() -> None:
    load_dotenv(find_repo_root() / ".env")
