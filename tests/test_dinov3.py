"""CPU-only checks for the DINO anomaly heatmap."""

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import fish_segmentation.dinov3 as d
from fish_segmentation.dinov3 import (
    compute_anomaly_heatmap,
    erase_regions,
    mask_from_detections,
    replace_masked_tokens,
    run_dino_erase_loop,
    run_dino_suppress_loop,
    segment_anomalies,
    suppress_regions,
)


def test_compute_anomaly_heatmap_shape_and_range():
    torch.manual_seed(0)
    tokens = F.normalize(torch.randn(30, 8), p=2, dim=-1)

    heatmap = compute_anomaly_heatmap(
        tokens,
        grid_shape=(5, 6),
        target_shape=(36, 30),
    )

    assert heatmap.shape == (30, 36)
    assert np.isfinite(heatmap).all()
    assert np.isclose(heatmap.min(), 0.0)
    assert np.isclose(heatmap.max(), 1.0)


def test_compute_anomaly_heatmap_accepts_bf16_tokens():
    """DINOv3 checkpoints load in bf16 on GPU; .numpy() needs an fp32 cast first."""
    torch.manual_seed(0)
    tokens = F.normalize(torch.randn(30, 8, dtype=torch.bfloat16), p=2, dim=-1)

    heatmap = compute_anomaly_heatmap(
        tokens,
        grid_shape=(5, 6),
        target_shape=(36, 30),
    )

    assert heatmap.shape == (30, 36)
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()


def test_dino_pass_scales_the_capped_view(monkeypatch):
    captured = {}

    def fake_heatmap(
        image,
        model,
        processor,
        scale,
        device,
        max_patches,
        score_mode="mean",
        knn_k=5,
        local_contrast_strength=0.0,
        local_contrast_sigma_pct=0.05,
    ):
        captured["scale"] = scale
        return np.zeros((image.height, image.width), dtype=np.float32)

    monkeypatch.setattr(d, "_compute_dino_heatmap", fake_heatmap)
    monkeypatch.setattr(
        d,
        "segment_anomalies",
        lambda score, **kwargs: np.zeros_like(score, dtype=np.uint8),
    )

    d._dino_pass(
        Image.new("RGB", (2048, 1152)),
        model=None,
        processor=None,
        long_side=1024,
        device="cpu",
        resolution_scale=4.0,
    )

    assert np.isclose(captured["scale"], 2.0)


def test_compute_anomaly_heatmap_raw_mode():
    torch.manual_seed(0)
    tokens = F.normalize(torch.randn(30, 8), p=2, dim=-1)

    raw = compute_anomaly_heatmap(
        tokens, grid_shape=(5, 6), target_shape=(36, 30), normalize=False
    )
    norm = compute_anomaly_heatmap(tokens, grid_shape=(5, 6), target_shape=(36, 30))

    assert raw.shape == (30, 36)
    # normalize=True is exactly the min-max of the raw map
    expected = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    assert np.allclose(norm, expected)


def test_tiled_heatmap_covers_the_native_image(monkeypatch):
    calls = []

    def fake_heatmap(
        image, model, processor, scale, device, normalize=True,
        score_mode="mean", knn_k=5,
    ):
        calls.append(image.size)
        return np.ones((image.height, image.width), dtype=np.float32)

    monkeypatch.setattr(d, "_run_dino_heatmap", fake_heatmap)
    image = Image.new("RGB", (1000, 700))
    heatmap = d._compute_dino_heatmap(
        image,
        model=None,
        processor=None,
        scale=2.0,
        device="cpu",
        max_patches=1024,
    )

    assert len(calls) > 1
    assert heatmap.shape == (700, 1000)
    # constant raw tiles -> no relative anomaly -> flat zero map
    assert np.allclose(heatmap, 0.0)
    assert all(d._patch_count(size, 2.0) <= 1024 for size in calls)


def test_tiled_heatmap_cancels_repeating_bias(monkeypatch):
    """A per-tile positional 'bowl' must not survive into the stitched heatmap."""

    def bowl(shape):
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
        r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        return 1.0 - 0.8 * np.clip(r, 0.0, 1.0)

    def fake_heatmap(
        image, model, processor, scale, device, normalize=True,
        score_mode="mean", knn_k=5,
    ):
        raw = bowl(np.asarray(image).shape[:2]).copy()
        # content-dependent anomaly: only the tile containing the bright block
        raw[np.asarray(image.convert("L")) > 200] += 3.0
        return raw

    monkeypatch.setattr(d, "_run_dino_heatmap", fake_heatmap)
    arr = np.full((538, 960, 3), 60, np.uint8)
    arr[120:140, 60:90] = 255
    heatmap = d._compute_dino_heatmap(
        Image.fromarray(arr), model=None, processor=None, scale=4.0, device="cpu"
    )

    # far from the anomaly the map is flat: the bowl/lattice was cancelled
    assert heatmap[400:, 700:].max() < 0.5
    # the injected anomaly survives the correction
    assert heatmap[120:140, 60:90].min() > 0.9


def test_tiled_heatmap_preserves_relative_scores(monkeypatch):
    """Tiles are stitched in raw score space, so per-tile maxima cannot grid."""
    calls = []

    def fake_heatmap(
        image, model, processor, scale, device, normalize=True,
        score_mode="mean", knn_k=5,
    ):
        calls.append(image.size)
        value = np.float32(0.9 if np.asarray(image).mean() > 127 else 0.2)
        if normalize:
            # old per-tile contract: a constant tile would normalize to zeros
            return np.zeros((image.height, image.width), dtype=np.float32)
        return np.full((image.height, image.width), value, dtype=np.float32)

    monkeypatch.setattr(d, "_run_dino_heatmap", fake_heatmap)

    arr = np.full((700, 1000, 3), 40, np.uint8)
    arr[:, 500:] = 255
    image = Image.fromarray(arr)
    heatmap = d._compute_dino_heatmap(
        image,
        model=None,
        processor=None,
        scale=2.0,
        device="cpu",
        max_patches=1024,
    )

    assert len(calls) > 1
    assert heatmap.shape == (700, 1000)
    # dark (raw 0.2) vs bright (raw 0.9) tiles stay separated after the stitch
    assert heatmap[:, :300].max() < 0.5
    assert heatmap[:, 700:].min() > 0.5


def test_score_modes_and_input_validation():
    torch.manual_seed(0)
    tokens = F.normalize(torch.randn(8 * 8, 16), p=2, dim=-1)

    for kwargs in (
        {"score_mode": "knn", "knn_k": 3},
        {"score_mode": "mean", "local_contrast_strength": 1.0},
    ):
        heat = compute_anomaly_heatmap(tokens, (8, 8), (32, 32), **kwargs)
        assert heat.shape == (32, 32)
        assert np.isfinite(heat).all()
        assert 0.0 <= heat.min() and heat.max() <= 1.0 + 1e-6

    for bad in (
        dict(score_mode="nope"),
        dict(knn_k=0),
        dict(local_contrast_strength=-1.0),
        dict(local_contrast_strength=1.0, local_contrast_sigma_pct=0.0),
    ):
        with pytest.raises(ValueError):
            compute_anomaly_heatmap(tokens, (8, 8), (32, 32), **bad)

    flat = np.full((16, 16), 0.5, np.float32)
    with pytest.raises(ValueError):
        segment_anomalies(flat, method="nope")
    with pytest.raises(ValueError):
        segment_anomalies(flat, percentile_threshold=97.0, low_percentile=99.0)
    with pytest.raises(ValueError):
        segment_anomalies(flat, closing_kernel_size=-1)


def test_grow_seeds_keeps_only_seeded_components():
    low = np.zeros((10, 30), np.uint8)
    low[2:5, 1:9] = 1  # component without seed
    low[2:5, 12:20] = 1  # component with seed
    low[3, 20:24] = 1  # thin bridge to a third area
    low[2:5, 24:29] = 1
    seeds = np.zeros((10, 30), np.uint8)
    seeds[3, 15] = 1

    grown = d._grow_seeds(seeds, low)

    assert grown[3, 5] == 0, "unseeded component must be dropped"
    assert grown[3, 15] == 1 and grown[3, 26] == 1, grown


def test_hysteresis_grows_through_a_weak_waist():
    """The percentile core splits the object; hysteresis keeps it as one region."""
    rng = np.random.default_rng(0)
    score = rng.uniform(0.0, 0.1, (200, 200)).astype(np.float32)
    score[90:110, 80:120] = 0.9  # small object (2% of the frame)
    score[98:102, 80:120] = 0.6  # waist: below the seeds, above the background

    core = segment_anomalies(score, percentile_threshold=99.5, low_percentile=99.5)
    grown = segment_anomalies(score, percentile_threshold=99.5, low_percentile=97.0)

    _, _, core_stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    _, _, grown_stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    core_big = [a for a in core_stats[1:, cv2.CC_STAT_AREA] if a > 100]
    grown_big = [a for a in grown_stats[1:, cv2.CC_STAT_AREA] if a > 100]

    assert len(core_big) == 2, core_big  # the two bands, split by the waist
    assert len(grown_big) == 1, grown_big  # grown through the waist
    assert core[100, 100] == 0 and grown[100, 100] == 255
    assert np.count_nonzero(grown) > np.count_nonzero(core)


def test_local_contrast_suppresses_the_pedestal():
    yy, xx = np.mgrid[0:128, 0:128]
    pedestal = np.exp(-(((yy - 64) / 50.0) ** 2 + ((xx - 64) / 50.0) ** 2) / 2)
    peak = np.exp(-(((yy - 64) / 5.0) ** 2 + ((xx - 64) / 5.0) ** 2) / 2)
    score = (0.4 * pedestal + peak).astype(np.float32)

    sharp = d._local_contrast(score, strength=1.0, sigma_pct=0.25)

    halo = (slice(2, 12), slice(2, 12))
    assert np.median(sharp[halo]) < 0.5 * np.median(score[halo])
    assert sharp.max() > 0.5, "the peak must survive the high-pass"


# ---------------------------------------------------------------------------
# Suppression, erase and rerun loop
# ---------------------------------------------------------------------------


def _unit(vector):
    return F.normalize(torch.tensor(vector, dtype=torch.float32), p=2, dim=-1)


def test_compute_anomaly_heatmap_suppress_mask_pins_and_pools():
    """A suppressed patch must not score, and the runner-up must take over."""
    base = _unit([1.0, 0.0, 0.0, 0.0])
    strongest = _unit([0.0, 1.0, 0.0, 0.0])  # orthogonal to base
    runner_up = _unit([1.0, 1.0, 0.0, 0.0])  # 45 degrees to base

    tokens = base.repeat(64, 1)  # 8x8 grid, 16 px per patch at target 128
    tokens[27] = strongest  # patch (3, 3), away from the suppressed border
    tokens[54] = runner_up  # patch (6, 6), far from the suppressed patch

    suppressed = np.zeros((8, 8), dtype=bool)
    suppressed[3, 3] = True

    # At grid resolution the pin and the pool swap are exact.
    grid_plain = compute_anomaly_heatmap(tokens, (8, 8), (8, 8))
    grid_rerun = compute_anomaly_heatmap(
        tokens, (8, 8), (8, 8), suppress_mask=suppressed
    )
    assert grid_plain[3, 3] == grid_plain.max()
    assert grid_rerun[3, 3] == grid_rerun.min(), "suppressed patch pinned to the min"
    assert grid_rerun[6, 6] == grid_rerun.max(), "runner-up takes over"

    plain = compute_anomaly_heatmap(tokens, (8, 8), (128, 128))
    rerun = compute_anomaly_heatmap(tokens, (8, 8), (128, 128), suppress_mask=suppressed)

    strongest_center = (slice(52, 60), slice(52, 60))
    runner_up_center = (slice(100, 108), slice(100, 108))
    assert plain[strongest_center].mean() > plain[runner_up_center].mean()
    # At pixel resolution bicubic ringing puts raw.min() below zero, so the final
    # min-max normalization lifts the pinned floor to ~0.1 (measured 0.1001):
    # absolute thresholds are meaningless here, compare against the survivor.
    assert rerun[strongest_center].max() < 0.25 * rerun[runner_up_center].mean()
    assert rerun[runner_up_center].mean() > 0.1, "runner-up must survive"

    with pytest.raises(ValueError):
        compute_anomaly_heatmap(tokens, (8, 8), (128, 128), suppress_mask=np.zeros((3, 3), bool))


def test_mask_to_patch_grid_marks_any_covered_patch():
    mask = np.zeros((32, 32), dtype=bool)
    mask[1, 30] = True  # single pixel in the top-right patch
    grids = d._mask_to_patch_grid(mask, (2, 2))

    assert grids[0, 1] and not grids[0, 0] and not grids[1, 0] and not grids[1, 1]
    # an already-patch-sized mask is returned as a copy
    patch_mask = np.zeros((2, 2), dtype=bool)
    patch_mask[1, 0] = True
    assert np.array_equal(d._mask_to_patch_grid(patch_mask, (2, 2)), patch_mask)


def test_replace_masked_tokens_uses_water_mean():
    tokens = _unit([1.0, 0.0, 0.0]).repeat(4, 1)
    tokens[1] = _unit([0.0, 1.0, 0.0])  # object patch
    patch_mask = np.array([False, True, False, False])

    replaced = replace_masked_tokens(tokens, patch_mask)

    assert torch.allclose(replaced[1], tokens[0]), "masked token must become the water prototype"
    assert torch.allclose(replaced[0], tokens[0]) and torch.allclose(replaced[2], tokens[2])
    assert torch.allclose(replaced.norm(dim=-1), torch.ones(4), atol=1e-5)

    # degenerate masks are a no-op
    assert replace_masked_tokens(tokens, np.zeros(4, bool)) is tokens
    assert replace_masked_tokens(tokens, np.ones(4, bool)) is tokens

    with pytest.raises(ValueError):
        replace_masked_tokens(tokens, np.zeros(3, bool))


def test_suppress_regions_pins_to_minimum():
    score = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True

    out = suppress_regions(score, mask)

    assert out[mask].max() == score.min()
    assert np.array_equal(out[~mask], score[~mask])
    assert out is not score


def test_segment_anomalies_valid_mask_keeps_weak_peak():
    rng = np.random.default_rng(0)
    score = rng.uniform(0.0, 0.1, (100, 100)).astype(np.float32)
    score[10:50, 10:50] = 0.9  # dominant blob (16% of the frame)
    score[70:90, 70:90] = 0.5  # weaker peak (4%)

    plain = segment_anomalies(score, percentile_threshold=99.5, low_percentile=98.0)
    valid = np.ones_like(score, dtype=bool)
    valid[10:50, 10:50] = False
    rerun = segment_anomalies(
        score, percentile_threshold=99.5, low_percentile=98.0, valid_mask=valid
    )

    assert plain[80, 80] == 0, "weak peak loses the percentile race against the blob"
    assert rerun[80, 80] == 255, "ignoring the blob lets the weak peak seed"
    assert rerun[10:50, 10:50].max() == 0, "suppressed pixels must not re-enter the mask"

    with pytest.raises(ValueError):
        segment_anomalies(score, valid_mask=np.zeros_like(score, dtype=bool))
    with pytest.raises(ValueError):
        segment_anomalies(score, valid_mask=np.ones((3, 3), dtype=bool))


def test_mask_from_detections_bbox_radius_and_dilation():
    det = {"x": 0.5, "y": 0.5, "radius": 0.05, "bbox": (0.4, 0.4, 0.6, 0.6)}

    box_mask = mask_from_detections([det], (100, 50))

    assert box_mask[20:30, 40:60].all()
    assert box_mask.sum() == 10 * 20

    grown = mask_from_detections([det], (100, 50), dilate_px=2)
    assert grown.sum() > box_mask.sum()

    circle = mask_from_detections([det], (100, 50), shape="radius")
    cy, cx = np.nonzero(circle)
    assert abs(cx.mean() - 50) < 2 and abs(cy.mean() - 25) < 2
    assert circle.sum() < box_mask.sum()

    no_box = mask_from_detections([{**det, "bbox": None}], (100, 50))
    assert no_box.sum() == circle.sum()


def test_erase_regions_inpaint_and_patch():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    arr[10:30, 20:40] = 0
    image = Image.fromarray(arr)
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 20:40] = True

    for mode in ("inpaint", "patch"):
        erased = erase_regions(image, mask, mode=mode)
        out = np.asarray(erased)
        assert not np.array_equal(out[mask], arr[mask]), mode
        assert np.array_equal(out[~mask], arr[~mask]), mode
        assert out.shape == arr.shape and out.dtype == np.uint8

    # a corner mask has a clean source patch, so 'patch' really pastes one
    corner = np.zeros((40, 60), dtype=bool)
    corner[2:12, 2:12] = True
    patched = np.asarray(erase_regions(image, corner, mode="patch"))
    assert not np.array_equal(patched[corner], arr[corner])
    assert np.array_equal(patched[~corner], arr[~corner])

    # an empty mask returns an untouched copy
    empty = erase_regions(image, np.zeros((40, 60), bool))
    assert np.array_equal(np.asarray(empty), arr)

    with pytest.raises(ValueError):
        erase_regions(image, mask, mode="nope")
    with pytest.raises(ValueError):
        erase_regions(image, np.zeros((10, 10), bool))


def test_dino_pass_forwards_suppression_as_valid_mask(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        d,
        "_compute_dino_heatmap",
        lambda image, model, processor, scale, device, **kwargs: np.zeros(
            (image.height, image.width), dtype=np.float32
        ),
    )

    def fake_segment(score, **kwargs):
        captured.update(kwargs)
        return np.zeros_like(score, dtype=np.uint8)

    monkeypatch.setattr(d, "segment_anomalies", fake_segment)

    suppress = np.zeros((20, 30), dtype=bool)
    suppress[5:10, 5:10] = True
    d._dino_pass(
        Image.new("RGB", (30, 20)),
        model=None,
        processor=None,
        long_side=1024,
        device="cpu",
        suppress_mask=suppress,
    )

    assert np.array_equal(captured["valid_mask"], ~suppress)


def _det(x, y, score=0.9, bbox=None, radius=0.05):
    if bbox is None:
        bbox = (x - 0.05, y - 0.05, x + 0.05, y + 0.05)
    return {
        "x": x,
        "y": y,
        "radius": radius,
        "bbox": bbox,
        "area": 100.0,
        "solidity": 0.8,
        "score_mean": score,
        "score_p95": score,
        "level": 0,
        "n_levels": 1,
    }


def test_run_dino_erase_loop_flags_leakage_and_stops(monkeypatch):
    a = _det(0.2, 0.2, score=0.9)
    b = _det(0.8, 0.8, score=0.8)
    c = _det(0.5, 0.5, score=0.7)
    per_call = [[a, b], [b, c], []]
    seen_images = []

    def fake_detector(image, model, processor, **kwargs):
        seen_images.append(image.copy())
        return per_call.pop(0)

    monkeypatch.setattr(d, "run_zoom_detector", fake_detector)

    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    )
    records = run_dino_erase_loop(
        image, model=None, processor=None, max_iterations=4, dilate_px=1
    )

    assert len(records) == 3, "must stop after the empty iteration"
    assert [det["x"] for det in records[0]["new_detections"]] == [0.2, 0.8]
    assert records[0]["leaked_detections"] == []
    # iteration 1 sees the erased frame: b is a leakage artifact, c is new
    assert [det["x"] for det in records[1]["leaked_detections"]] == [0.8]
    assert [det["x"] for det in records[1]["new_detections"]] == [0.5]
    assert records[1]["new_detections"][0]["iteration"] == 1
    assert records[0]["mask"].any() and records[1]["mask"].any()
    assert not records[2]["new_detections"] and not records[2]["mask"].any()
    assert not np.array_equal(np.asarray(seen_images[0]), np.asarray(seen_images[1]))
    # accumulated mask covers both erased iterations
    assert np.array_equal(
        records[2]["suppressed_mask"], records[1]["accumulated_mask"]
    )
    assert records[2]["suppressed_mask"].sum() >= records[1]["mask"].sum()


def test_run_dino_erase_loop_mask_fn_and_validation(monkeypatch):
    det = _det(0.5, 0.5)
    calls = []

    def fake_detector(image, model, processor, **kwargs):
        calls.append(1)
        return [det] if len(calls) == 1 else []

    monkeypatch.setattr(d, "run_zoom_detector", fake_detector)

    def mask_fn(detections, image):
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[:10, :10] = True
        return mask

    image = Image.fromarray(np.full((50, 50, 3), 120, np.uint8))
    records = run_dino_erase_loop(
        image, model=None, processor=None, mask_fn=mask_fn, max_iterations=2
    )
    assert records[0]["mask"][:10, :10].all()

    monkeypatch.setattr(
        d, "run_zoom_detector", lambda image, model, processor, **kwargs: [det]
    )
    with pytest.raises(ValueError):
        run_dino_erase_loop(
            image,
            model=None,
            processor=None,
            mask_fn=lambda detections, img: np.zeros((3, 3), bool),
            max_iterations=1,
        )
    with pytest.raises(ValueError):
        run_dino_erase_loop(image, model=None, processor=None, max_iterations=0)


def test_run_dino_suppress_loop_accumulates_masks(monkeypatch):
    det = _det(0.5, 0.5)
    calls = []
    suppress_kwargs = []

    def fake_detector(image, model, processor, **kwargs):
        calls.append(1)
        suppress_kwargs.append(kwargs.get("suppress_mask"))
        return [det] if len(calls) == 1 else []

    monkeypatch.setattr(d, "run_zoom_detector", fake_detector)

    image = Image.fromarray(np.full((60, 60, 3), 100, np.uint8))
    records = run_dino_suppress_loop(
        image,
        model=None,
        processor=None,
        suppress_mode="tokens",
        max_iterations=3,
        dilate_px=0,
    )

    assert len(records) == 2
    assert suppress_kwargs[0] is None, "the baseline pass is not suppressed"
    assert suppress_kwargs[1] is not None and suppress_kwargs[1][30, 30]
    assert records[1]["suppressed_mask"][30, 30]
    assert [det["x"] for det in records[0]["new_detections"]] == [0.5]
    with pytest.raises(ValueError):
        run_dino_suppress_loop(image, model=None, processor=None, max_iterations=0)
