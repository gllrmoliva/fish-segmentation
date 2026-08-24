import gdown
import os
from pathlib import Path

path = Path(__file__).parent.parent / "dataset"


folder_id = "1gWVwK3860mkuir6n-GpxoeOXUtKIXy9i"
url = f"https://drive.google.com/drive/folders/{folder_id}"
os.makedirs(path,
            exist_ok=True)

print(f"Downloading to {path}...")

gdown.download_folder(
    url,
    output=str(path),
    quiet=False,
    use_cookies=False
)

print("Ready!")
