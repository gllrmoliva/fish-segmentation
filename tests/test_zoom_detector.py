"""Self-check for zoom detector geometry — no GPU, no model, mocked DINO pass."""
import numpy as np
from PIL import Image

from fish_segmentation.dinov3 import (
    _dino_pass,
    _interest_zones,
    _merge_points,
    run_zoom_detector,
)


def test_interest_zones():
    mask = np.zeros((100, 200), np.uint8)
    mask[10:30, 20:60] = 255  # one blob (800 px)
    mask[10:30, 150:190] = 255  # another, far away
    mask[80:82, 100:102] = 255  # speck (4 px), should be dropped
    zones = _interest_zones(mask, margin=0.0, min_area=30)
    assert len(zones) == 2, zones
    # margin expansion + clamped to image bounds: 40x20 blob, margin 0.5 -> +20/+10
    zones_m = _interest_zones(mask, margin=0.5, min_area=30)
    x0, y0, x1, y1 = zones_m[0]
    assert (x0, y0, x1, y1) == (0, 0, 80, 40), zones_m[0]
    # two near blobs whose expanded boxes overlap get merged into one zone
    near = np.zeros((100, 200), np.uint8)
    near[10:20, 10:20] = 255
    near[24:34, 24:34] = 255
    assert len(_interest_zones(near, margin=0.6, min_area=30)) == 1
    print("zones ok:", zones, zones_m)


def test_merge_points():
    # same object seen at level 1 (finer) and level 0 (coarse) -> one point, finer wins
    pts = [(100, 100, 10.0, 0), (104, 102, 20.0, 1)]
    merged = _merge_points(pts)
    assert len(merged) == 1, merged
    x, y, r, lvl = merged[0]
    assert (x, y) == (102, 101) and r == 20.0 and lvl == 1, merged[0]
    # two distant objects stay separate
    assert len(_merge_points([(0, 0, 5, 0), (500, 500, 5, 0)])) == 2
    print("merge ok:", merged)


def test_zoom_roundtrip(monkey_img_size=(2048, 1152)):
    """Fake DINO: blob in the middle of the full frame. Zoom pipeline must find it
    at normalized ~ (0.5, 0.5) despite intermediate resizes."""
    W, H = monkey_img_size
    calls = []

    def fake_dino_pass(image, model, processor, long_side, device, percentile_threshold=99.5):
        w, h = image.size
        calls.append((image.size, long_side))
        mask = np.zeros((h, w), np.uint8)
        # blob at normalized (0.5, 0.5) of THIS view
        cx, cy = w // 2, h // 2
        mask[cy - 5 : cy + 5, cx - 5 : cx + 5] = 255
        return mask

    import fish_segmentation.dinov3 as d

    orig = d._dino_pass
    d._dino_pass = fake_dino_pass
    try:
        img = Image.new("RGB", (W, H))
        results = run_zoom_detector(img, model=None, processor=None, long_side=1024)
    finally:
        d._dino_pass = orig

    assert calls, "no dino passes"
    # first view: full frame, fed downscaled to long_side 1024
    assert calls[0] == ((W, H), 1024), calls[0]
    # level-0 blob found -> one zoom -> native 16x16 crop <= 1024 -> recursion stops
    assert len(calls) == 2 and max(calls[1][0]) <= 1024, calls
    # zoom happened (>= 2 levels) OR stopped because view was native<=1024
    print(f"{len(calls)} passes; sizes: {[s for s, _ in calls]}")
    # center-of-frame blob maps back to normalized ~ (0.5, 0.5)
    best = min(results, key=lambda p: (p["x"] - 0.5) ** 2 + (p["y"] - 0.5) ** 2)
    assert abs(best["x"] - 0.5) < 0.02 and abs(best["y"] - 0.5) < 0.02, best
    assert 0 <= best["x"] <= 1 and 0 <= best["y"] <= 1
    print("roundtrip ok:", best, f"({len(results)} points total)")


if __name__ == "__main__":
    test_interest_zones()
    test_merge_points()
    test_zoom_roundtrip()
    print("ALL OK")
