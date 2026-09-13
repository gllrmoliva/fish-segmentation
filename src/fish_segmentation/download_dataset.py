from pathlib import Path

import gdown

from fish_segmentation.paths import find_repo_root


def main() -> None:
    path = find_repo_root() / "dataset"
    folder_id = "1gWVwK3860mkuir6n-GpxoeOXUtKIXy9i"
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to {path}...")

    gdown.download_folder(
        url,
        output=str(path),
        quiet=False,
        use_cookies=False
    )

    print("Ready!")


if __name__ == "__main__":
    main()
