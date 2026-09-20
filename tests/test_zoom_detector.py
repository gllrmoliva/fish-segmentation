"""Self-check for zoom detector geometry — no GPU, no model, mocked DINO pass."""
import numpy as np
from PIL import Image

from fish_segmentation.dinov3 import (
    RawPoint,
    _dino_pass,
    _interest_zones,
    _merge_points,
    extract_salient_regions,
    letterbox_box,
    remove_letterbox,
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


def test_interest_zones_merge_gap():
    """Boxes closer than merge_gap fuse into one zone; a small gap keeps them apart."""
    mask = np.zeros((200, 200), np.uint8)
    mask[40:60, 20:40] = 255  # box (20, 40, 40, 60)
    mask[40:60, 80:100] = 255  # box (80, 40, 100, 60) — 40 px away

    merged = _interest_zones(mask, margin=0.0, merge_gap=0.25)
    assert merged == [(20, 40, 100, 60)], merged
    split = _interest_zones(mask, margin=0.0, merge_gap=0.1)
    assert len(split) == 2, split
    print("merge gap ok:", merged, split)


def test_interest_zones_min_size():
    """A tiny component yields a context crop of at least min_zone_fraction."""
    mask = np.zeros((200, 200), np.uint8)
    mask[95:105, 95:105] = 255

    zones = _interest_zones(mask, margin=0.0, min_zone_fraction=0.5)
    assert len(zones) == 1, zones
    x0, y0, x1, y1 = zones[0]
    assert x1 - x0 >= 100 and y1 - y0 >= 100, zones
    assert x0 <= 95 and y0 <= 95 and x1 >= 105 and y1 >= 105, zones
    assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 200, zones
    print("min size ok:", zones)


def test_merge_points():
    # Same object at level 1 (finer) and level 0 (coarse): the deep region wins
    # and the coarse one is only absorbed as confirmation (no averaging).
    pts = [RawPoint(100, 100, 10.0, 0), RawPoint(104, 102, 20.0, 1)]
    merged = _merge_points(pts)
    assert len(merged) == 1, merged
    det = merged[0]
    assert (det["x"], det["y"]) == (104, 102), det  # real deep center, not (102, 101)
    assert det["radius"] == 20.0 and det["level"] == 1
    assert det["n_levels"] == 2 and det["n_points"] == 2 and det["levels"] == (0, 1)

    # a shallow detection matching no deeper cluster survives on its own
    far = [RawPoint(0, 0, 5.0, 0), RawPoint(500, 500, 5.0, 1)]
    merged_far = _merge_points(far)
    assert len(merged_far) == 2, merged_far
    assert all(d["n_levels"] == 1 for d in merged_far)

    # same-level duplicates cluster around the largest inscribed circle
    same = [RawPoint(100, 100, 5.0, 1), RawPoint(103, 100, 12.0, 1)]
    merged_same = _merge_points(same)
    assert len(merged_same) == 1 and merged_same[0]["radius"] == 12.0, merged_same

    # a wide coarse region confirms a deep point even when its inscribed circle
    # does not reach it (the center of a hollow/elongated blob can sit far away)
    wide = [
        RawPoint(100, 100, 5.0, 0, bbox=(0, 0, 220, 200)),
        RawPoint(200, 100, 10.0, 1, bbox=(190, 90, 210, 110)),
    ]
    merged_wide = _merge_points(wide)
    assert len(merged_wide) == 1, merged_wide
    assert merged_wide[0]["level"] == 1 and merged_wide[0]["n_levels"] == 2, merged_wide

    # deterministic under permutation
    assert _merge_points([pts[1], pts[0]]) == _merge_points(pts)

    # A/B mode: the absorbed coarse detection is emitted as well
    kept = _merge_points(pts, keep_absorbed=True)
    assert len(kept) == 2, kept
    shallow = [d for d in kept if d["level"] == 0]
    assert len(shallow) == 1
    assert (shallow[0]["x"], shallow[0]["y"], shallow[0]["radius"]) == (100, 100, 10.0)
    print("merge ok:", merged, merged_far)


def test_extract_salient_regions_metadata():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:20, 10:20] = 255  # 10x10 square
    score = np.zeros((40, 40), np.float32)
    score[10:20, 10:20] = 0.8

    _, regions = extract_salient_regions(mask, score_map=score)
    assert len(regions) == 1, regions
    reg = regions[0]
    assert reg["bbox"] == (10, 10, 20, 20), reg
    assert reg["area"] == 100
    # 10x10 square: r=5 -> area/(pi r^2) = 4/pi
    assert abs(reg["solidity"] - 4 / np.pi) < 1e-4, reg["solidity"]
    assert abs(reg["score_mean"] - 0.8) < 1e-6
    assert abs(reg["score_p95"] - 0.8) < 1e-6

    _, plain = extract_salient_regions(mask)
    assert "score_mean" not in plain[0]
    print("region metadata ok:", reg)


def test_letterbox_box():
    arr = np.zeros((80, 100, 3), np.uint8)
    arr[10:60, 20:90] = 255
    boxed = Image.fromarray(arr)
    assert letterbox_box(boxed) == (20, 10, 90, 60)
    assert remove_letterbox(boxed).size == (70, 50)
    # all-black falls back to the full image
    assert letterbox_box(Image.new("RGB", (10, 10))) == (0, 0, 10, 10)
    print("letterbox ok")


def test_zoom_detector_maps_points_to_the_original_frame():
    """Detections are normalized to the ORIGINAL frame (letterbox offset included)."""
    arr = np.zeros((120, 200, 3), np.uint8)
    arr[30:110, 50:180] = 100  # content box (50, 30, 180, 110) -> 130x80
    image = Image.fromarray(arr)

    def fake_dino_pass(image, model, processor, long_side, device, **kwargs):
        w, h = image.size
        mask = np.zeros((h, w), np.uint8)
        mask[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5] = 255
        return mask, np.zeros((h, w), np.float32)

    import fish_segmentation.dinov3 as d

    orig = d._dino_pass
    d._dino_pass = fake_dino_pass
    try:
        detections = run_zoom_detector(image, model=None, processor=None, long_side=1024)
    finally:
        d._dino_pass = orig

    assert len(detections) == 1, detections
    det = detections[0]
    # minMaxLoc breaks the 2x2 center tie at the top-left pixel: (64, 39) in view coords
    assert abs(det["x"] - (50 + 64) / 200) < 1e-6, det
    assert abs(det["y"] - (30 + 39) / 120) < 1e-6, det
    assert abs(det["radius"] - 5 / 200) < 1e-6, det
    assert det["level"] == 0 and det["n_levels"] == 1
    bx0, by0, _, _ = det["bbox"]
    assert abs(bx0 - (50 + 60) / 200) < 1e-6 and abs(by0 - (30 + 35) / 120) < 1e-6, det
    print("original-frame mapping ok:", det)


def test_zoom_min_confirmations_filter():
    """A coarse-only blob is dropped by min_confirmations=2; the refined one stays."""
    arr = np.full((1152, 2048, 3), 40, np.uint8)
    arr[576 - 60 : 576 + 60, 1024 - 60 : 1024 + 60] = 200
    image = Image.fromarray(arr)

    def fake_dino_pass(image, model, processor, long_side, device, **kwargs):
        w, h = image.size
        mask = np.zeros((h, w), np.uint8)
        if max(w, h) > 1024:  # root view: a center blob (will be refined) ...
            mask[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5] = 255
            mask[5:15, 5:15] = 255  # ... and a corner blob (no deep match)
        elif np.asarray(image.convert("L")).mean() > 100:  # bright center crop
            mask[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5] = 255
        return mask, np.zeros((h, w), np.float32)

    import fish_segmentation.dinov3 as d

    orig = d._dino_pass
    d._dino_pass = fake_dino_pass
    try:
        all_dets = run_zoom_detector(
            image, model=None, processor=None, long_side=1024, min_confirmations=1
        )
        confirmed = run_zoom_detector(
            image, model=None, processor=None, long_side=1024, min_confirmations=2
        )
    finally:
        d._dino_pass = orig

    assert len(all_dets) == 2, all_dets
    assert sorted(d["n_levels"] for d in all_dets) == [1, 2], all_dets
    assert len(confirmed) == 1, confirmed
    assert confirmed[0]["n_levels"] == 2, confirmed
    assert abs(confirmed[0]["x"] - 0.5) < 0.02 and abs(confirmed[0]["y"] - 0.5) < 0.02
    print("min_confirmations ok:", confirmed)


def test_zoom_trace_records_views():
    """trace= collects one record per view, in depth-first order, fully populated."""
    W, H = 2048, 1152
    seen = []

    def fake_dino_pass(
        image,
        model,
        processor,
        long_side,
        device,
        percentile_threshold=99.5,
        resolution_scale=1.0,
        max_patches=4096,
        options=None,
        trace=None,
    ):
        w, h = image.size
        seen.append(image.size)
        if trace is not None:
            trace["scale"] = 2.0
            trace["norm_score"] = np.zeros((h, w), np.float32)
        mask = np.zeros((h, w), np.uint8)
        mask[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5] = 255
        return mask, np.zeros((h, w), np.float32)

    import fish_segmentation.dinov3 as d

    orig = d._dino_pass
    d._dino_pass = fake_dino_pass
    try:
        trace = []
        results = run_zoom_detector(
            Image.new("RGB", (W, H)),
            model=None,
            processor=None,
            long_side=1024,
            resolution_scale=4.0,
            trace=trace,
        )
    finally:
        d._dino_pass = orig

    assert len(trace) == len(seen) == 2, (trace, seen)
    assert trace[0]["depth"] == 0 and trace[0]["origin"] == (0, 0)
    assert trace[0]["size"] == (W, H) and trace[1]["depth"] == 1
    assert trace[0]["feed_native"] is False and trace[1]["feed_native"] is True
    assert len(trace[0]["zones"]) == 1 and trace[1]["zones"] == []
    for entry in trace:
        assert entry["scale"] == 2.0
        assert entry["norm_score"].shape == (entry["size"][1], entry["size"][0])
        assert entry["mask"].shape == entry["norm_score"].shape
        assert entry["regions"] and entry["points"]
        assert entry["points"][0].level == entry["depth"]
        assert entry["points"][0].bbox is not None
    assert results, results
    print("trace ok:", [entry["size"] for entry in trace], f"{len(results)} point(s)")


def test_zoom_roundtrip(monkey_img_size=(2048, 1152)):
    """Fake DINO: blob in the middle of the full frame. Zoom pipeline must find it
    at normalized ~ (0.5, 0.5) despite intermediate resizes."""
    W, H = monkey_img_size
    calls = []

    def fake_dino_pass(
        image,
        model,
        processor,
        long_side,
        device,
        percentile_threshold=99.5,
        resolution_scale=1.0,
        max_patches=4096,
        options=None,
    ):
        w, h = image.size
        calls.append((image.size, long_side, resolution_scale, max_patches))
        mask = np.zeros((h, w), np.uint8)
        # blob at normalized (0.5, 0.5) of THIS view
        cx, cy = w // 2, h // 2
        mask[cy - 5 : cy + 5, cx - 5 : cx + 5] = 255
        return mask, np.zeros((h, w), np.float32)

    import fish_segmentation.dinov3 as d

    orig = d._dino_pass
    d._dino_pass = fake_dino_pass
    try:
        img = Image.new("RGB", (W, H))
        results = run_zoom_detector(
            img,
            model=None,
            processor=None,
            long_side=1024,
            resolution_scale=4.0,
        )
    finally:
        d._dino_pass = orig

    assert calls, "no dino passes"
    # first view: full frame, fed downscaled to long_side 1024
    assert calls[0] == ((W, H), 1024, 4.0, 4096), calls[0]
    # level-0 blob found -> one zoom -> native 16x16 crop <= 1024 -> recursion stops
    assert len(calls) == 2 and max(calls[1][0]) <= 1024, calls
    # zoom happened (>= 2 levels) OR stopped because view was native<=1024
    print(f"{len(calls)} passes; sizes: {[call[0] for call in calls]}")
    # center-of-frame blob maps back to normalized ~ (0.5, 0.5)
    best = min(results, key=lambda p: (p["x"] - 0.5) ** 2 + (p["y"] - 0.5) ** 2)
    assert abs(best["x"] - 0.5) < 0.02 and abs(best["y"] - 0.5) < 0.02, best
    assert 0 <= best["x"] <= 1 and 0 <= best["y"] <= 1
    print("roundtrip ok:", best, f"({len(results)} points total)")


if __name__ == "__main__":
    test_interest_zones()
    test_interest_zones_merge_gap()
    test_interest_zones_min_size()
    test_merge_points()
    test_extract_salient_regions_metadata()
    test_letterbox_box()
    test_zoom_detector_maps_points_to_the_original_frame()
    test_zoom_min_confirmations_filter()
    test_zoom_trace_records_views()
    test_zoom_roundtrip()
    print("ALL OK")
