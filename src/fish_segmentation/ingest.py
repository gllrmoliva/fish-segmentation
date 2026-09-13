import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import ffmpeg
import exiftool


def process_video(
    input_path: str,
    output_path: str,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    start_time: float | None = None,
    duration: float | None = None,
    fps: int | float | None = None,
    crf: int = 18,
) -> None:
    """
    adjust resolution, FPS, start offset, and duration

    :param input_path: Path to the source video file.
    :param output_path: Path for the exported video file.
    :param target_width: Target canvas width in pixels. If None (with
        target_height), the original resolution is kept untouched.
    :param target_height: Target canvas height in pixels.
    :param start_time: Start offset in seconds (optional).
    :param duration: Total length to extract in seconds from start_time (optional).
    :param fps: Desired output frame rate (optional).
    :param crf: x264 quality (0 = lossless, 18 = visually lossless).
    """
    input_kwargs = {}

    if start_time is not None:
        input_kwargs["ss"] = start_time

    if duration is not None:
        input_kwargs["t"] = duration

    stream = ffmpeg.input(input_path, **input_kwargs)

    # separate video stream
    video = stream.video

    if target_width is not None and target_height is not None:
        # scale maintaining aspect ratio (black bars)
        video = video.filter(
            "scale",
            w=target_width,
            h=target_height,
            force_original_aspect_ratio="decrease",
        ).filter(
            "pad",
            w=target_width,
            h=target_height,
            x="(ow-iw)/2",
            y="(oh-ih)/2",
            color="black",
        )

    # apply frame rate filter if specified
    if fps is not None:
        video = video.filter("fps", fps=fps)

    output_kwargs = {
        "vcodec": "libx264",
        "acodec": "aac",
        "pix_fmt": "yuv420p",
        "crf": crf,
    }

    output = ffmpeg.output(video, output_path, **output_kwargs)

    # execute
    ffmpeg.run(output, overwrite_output=True)


def extract_datetime_from_filename(filename: str) -> Optional[str]:
    """Extracts timestamp from DJI pattern: DJI_YYYYMMDDHHMMSS_..."""
    match = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = match.groups()
        return f"{year}{month}{day}_{hour}{minute}"
    return None


def get_recording_datetime_and_metadata(file_path: Path) -> tuple[str, dict]:
    """extracts metadata using ExifTool, fallback regex, and FFprobe."""
    resolved_path = str(file_path.resolve())
    stat = file_path.stat()

    metadata = {
        "file_system": {
            "filename": file_path.name,
            "original_path": resolved_path,
            "size_bytes": stat.st_size,
            "filesystem_mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": file_path.suffix.lower()
        },
        "exif_metadata": {},
        "ffprobe_streams": {}
    }

    # 1. Read metadata (ExifTool)
    try:
        with exiftool.ExifToolHelper() as et:
            raw_exif = et.get_metadata(resolved_path)
            if raw_exif:
                metadata["exif_metadata"] = raw_exif[0]
    except Exception as e:
        metadata["exif_metadata"] = {"error": str(e)}

    # read streams (FFprobe)
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        resolved_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metadata["ffprobe_streams"] = json.loads(result.stdout)
    except Exception as e:
        metadata["ffprobe_streams"] = {"error": str(e)}

    # true recording timestamp
    exif = metadata["exif_metadata"]
    date_str = None

    # Check EXIF/QuickTime internal date tags
    candidate_keys = [
        "QuickTime:CreationDate",
        "QuickTime:CreateDate",
        "EXIF:DateTimeOriginal",
        "XMP:CreateDate",
        "XMP:DateTimeOriginal"
    ]

    for key in candidate_keys:
        raw_val = exif.get(key)
        if raw_val and isinstance(raw_val, str) and not raw_val.startswith("0000"):
            # strip trailing timezone offset ("+02:00", "-0400") before digit-stripping
            clean_date = re.sub(r"[+-]\d{2}:?\d{2}$", "", raw_val.strip())
            clean_date = re.sub(r"[^\d]", "", clean_date)[:14]  # YYYYMMDDhhmmss
            if len(clean_date) == 14:
                date_str = f"{clean_date[:8]}_{clean_date[8:]}"
                break

    if not date_str:  # extract from filename
        date_str = extract_datetime_from_filename(file_path.name)

    if not date_str:  # modified time on disk
        date_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y%m%d_%H%M%S")

    metadata["recording_timestamp"] = date_str
    return date_str, metadata


def generate_unique_folder_name(file_path: Path, recording_timestamp: str) -> str:
    """recording timestamp with short hash"""
    hash_str = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{recording_timestamp}_{hash_str}"


def process_video_dataset(input_dir: str, output_base_dir: str) -> None:
    """recursively traverses input_dir, ignores non-videos, extracts real metadata, and runs process_video."""

    input_folder = Path(input_dir)
    output_folder = Path(output_base_dir)

    valid_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".ts"}

    # search all subdirectories
    video_files = [
        f for f in input_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]

    print(f"Found {len(video_files)} video(s) to process.")

    for video_path in video_files:
        # metadata
        recording_date, full_metadata = get_recording_datetime_and_metadata(video_path)

        # output directory
        unique_folder_name = generate_unique_folder_name(video_path, recording_date)
        dest_dir = output_folder / unique_folder_name

        # output paths
        output_video_path = dest_dir / "video.mp4"
        metadata_path = dest_dir / "metadata.json"

        if output_video_path.exists():
            print(f"[skip] Already processed: {video_path.relative_to(input_folder)}")
            continue

        print(f"\n[+] Processing: {video_path.relative_to(input_folder)}")
        print(f"    Recorded on: {recording_date} -> Output: {dest_dir.name}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        process_video(
            input_path=str(video_path),
            output_path=str(output_video_path),
            fps=15,
        )

        # sidecar written only after a successful transcode
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=4, ensure_ascii=False)
