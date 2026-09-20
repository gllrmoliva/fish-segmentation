"""CPU-only checks for single-frame video loading (no model, no GPU)."""

import cv2
import numpy as np
import pytest
from PIL import Image

from fish_segmentation.video_io import load_frame_rgb


def _write_tiny_video(path, colors):
    # 4x4: the mp4v encoder rounds odd dimensions down, so keep both axes even
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (4, 4))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4 in this environment")
    for rgb in colors:
        writer.write(cv2.cvtColor(np.full((4, 4, 3), rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def test_load_frame_rgb_reads_one_frame(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

    frame = load_frame_rgb(str(video_path), 1)

    assert frame is not None
    assert frame.shape == (4, 4, 3)
    # mp4 is lossy, but a solid green frame must stay closer to green than to red/blue
    assert int(frame[..., 1].mean()) > int(frame[..., 0].mean())


def test_load_frame_rgb_out_of_range(tmp_path):
    video_path = tmp_path / "tiny.mp4"
    _write_tiny_video(video_path, [(255, 0, 0), (0, 255, 0)])

    assert load_frame_rgb(str(video_path), -1) is None
    assert load_frame_rgb(str(video_path), 999) is None


def test_load_frame_rgb_reads_jpeg_folders(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for idx, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        Image.new("RGB", (4, 3), color).save(frame_dir / f"{idx}.jpg")

    frame = load_frame_rgb(str(frame_dir), 1)

    assert frame is not None
    assert frame.shape == (3, 4, 3)
    assert load_frame_rgb(str(frame_dir), 2) is None
