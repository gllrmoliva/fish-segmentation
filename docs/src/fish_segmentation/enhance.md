# fish_segmentation.enhance

## Purpose

Marine video enhancement for aerial drone footage: removes surface glints/sun
glints, smooths while preserving edges, and boosts local contrast. Produces a
"clean" MP4 to feed the SAM3/DINOv3 notebooks (the glint inpainting especially
helps the anomaly detector, which otherwise fires on specular highlights).

Driver: `notebooks/02_enhance.ipynb`.

## Functions

```python
process_marine_video_ffmpeg(
    input_video_path, output_video_path,
    fps=15.0,                 # fallback only
    glint_threshold=235,      # HSV V-channel threshold for glint mask
    clahe_clip_limit=2.5,
    clahe_tile_grid_size=(8, 8),
)
```

Per-frame steps:

1. **Glint mask + inpainting** — threshold HSV V channel at `glint_threshold`,
   dilate 3x3 ellipse, `cv2.inpaint(..., INPAINT_TELEA, radius=3)`.
2. **Edge-preserving smoothing** — `cv2.edgePreservingFilter(RECURS_FILTER, sigma_s=50, sigma_r=0.4)`.
3. **CLAHE** on the LAB L channel only.
4. Write frames via `imageio.get_writer(..., codec="libx264", output_params=["-crf", "18", "-pix_fmt", "yuv420p"])`
   (imageio[ffmpeg/pyav] dependency), BGR→RGB before writing.

## Gotchas

- The `fps` argument is effectively ignored for normal videos: a valid source FPS
  overrides it. It only applies as a fallback when the source reports 0/NaN.
- `cv2.VideoCapture` + imageio writer is a frame-by-frame Python loop — slow on
  long clips; budget time for full DJI videos.
- Uses OpenCV BGR internally; output written correctly as RGB.
