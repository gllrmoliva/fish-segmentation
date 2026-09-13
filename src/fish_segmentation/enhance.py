from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
from tqdm import tqdm


def process_marine_video_ffmpeg(
    input_video_path: str,
    output_video_path: str,
    fps: float = 15.0,
    glint_threshold: int = 235,
    clahe_clip_limit: float = 2.5,
    clahe_tile_grid_size: tuple = (8, 8),
) -> None:
    """Processes marine aerial video frames and encodes them cleanly using FFmpeg."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_video_path}")

    try:
        # Read total frames for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0 and not np.isnan(video_fps):
            fps = video_fps

        # Pre-allocate CLAHE & structuring element
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        print(f"Enhancing video via FFmpeg ({total_frames} frames @ {fps} FPS)...")

        with imageio.get_writer(
            output_video_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        ) as writer:
            for _ in tqdm(range(max(1, total_frames))):
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Glint & foam mask + inpainting
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                v_channel = hsv[:, :, 2]
                _, glint_mask = cv2.threshold(
                    v_channel, glint_threshold, 255, cv2.THRESH_BINARY
                )
                glint_mask = cv2.dilate(glint_mask, kernel, iterations=1)
                deglinted = cv2.inpaint(
                    frame, glint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
                )

                # 2. Fast edge-preserving smoothing
                smoothed = cv2.edgePreservingFilter(
                    deglinted, flags=cv2.RECURS_FILTER, sigma_s=50, sigma_r=0.4
                )

                # 3. CLAHE on LAB Lightness channel
                lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_enhanced = clahe.apply(l)
                enhanced_lab = cv2.merge((l_enhanced, a, b))
                processed_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

                # Convert BGR (OpenCV) -> RGB (FFmpeg expects RGB)
                processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

                writer.append_data(processed_rgb)
    finally:
        cap.release()
    print(f"\nDone! Clean, uncorrupted MP4 saved to: {output_video_path}")
