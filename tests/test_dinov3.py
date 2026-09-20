"""CPU-only checks for the DINO anomaly heatmap."""

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import fish_segmentation.dinov3 as d
from fish_segmentation.dinov3 import compute_anomaly_heatmap, segment_anomalies


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
