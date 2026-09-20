import glob
import os
from typing import Union

import cv2
import numpy as np
from PIL import Image


def _sorted_jpgs(path: str) -> list[str]:
    """Glob `path/*.jpg`, sorted by frame index when names are "<frame_index>.jpg"."""
    paths = glob.glob(os.path.join(path, "*.jpg"))
    if not paths:
        return []
    try:
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {paths[:5]=}, '
            f"falling back to lexicographic sort."
        )
        paths.sort()
    return paths


def load_all_frames_rgb(
    video_path: Union[str, "os.PathLike"],
) -> Union[list[np.ndarray], list[str]]:
    """Load all frames of a video for visualization purposes (they are not used by the model).

    Returns RGB numpy arrays for MP4 input, or sorted JPEG paths for a frame folder.
    """
    if str(video_path).lower().endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames_for_vis = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return video_frames_for_vis

    return _sorted_jpgs(str(video_path))


def load_frame_rgb(
    video_path: Union[str, "os.PathLike"], frame_index: int
) -> Union[np.ndarray, None]:
    """Load one frame as an RGB numpy array (MP4 input or JPEG frame folder).

    Returns None when ``frame_index`` is negative or past the last frame. Used by
    the incremental DINO→SAM3 loop to read a single keyframe from the session
    proxy instead of loading the whole video.
    """
    if str(video_path).lower().endswith(".mp4"):
        cap = cv2.VideoCapture(str(video_path))
        if frame_index < 0:
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_paths = _sorted_jpgs(str(video_path))
    if frame_index < 0 or frame_index >= len(image_paths):
        return None
    return np.asarray(Image.open(image_paths[frame_index]).convert("RGB"))


def load_sampled_frames(video_path: str, n_frames: int) -> list[Image.Image]:
    """Load `n_frames` equally spaced as PIL Images."""
    if str(video_path).lower().endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        frame_indices = set(
            np.linspace(0, total_frames - 1, num=n_frames, dtype=int)
        )

        frames = []
        current_idx = 0
        while cap.isOpened() and len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in frame_indices:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # NumPy array to PIL Image
                frames.append(Image.fromarray(rgb_frame))
            current_idx += 1

        cap.release()
        return frames

    image_paths = _sorted_jpgs(video_path)
    if not image_paths:
        return []

    sample_indices = np.linspace(
        0, len(image_paths) - 1, num=min(n_frames, len(image_paths)), dtype=int
    )
    return [
        Image.open(image_paths[idx]).convert("RGB") for idx in sample_indices
    ]
