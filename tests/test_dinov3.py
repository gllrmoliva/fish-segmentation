"""CPU-only checks for the DINO anomaly heatmap."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import fish_segmentation.dinov3 as d
from fish_segmentation.dinov3 import compute_anomaly_heatmap


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


def test_dino_pass_scales_the_capped_view(monkeypatch):
    captured = {}

    def fake_heatmap(image, model, processor, scale, device, max_patches):
        captured["scale"] = scale
        return np.zeros((image.height, image.width), dtype=np.float32)

    monkeypatch.setattr(d, "_compute_dino_heatmap", fake_heatmap)
    monkeypatch.setattr(
        d,
        "segment_anomalies",
        lambda score, percentile_threshold=99.5: np.zeros_like(score, dtype=np.uint8),
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

    def fake_heatmap(image, model, processor, scale, device, normalize=True):
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

    def fake_heatmap(image, model, processor, scale, device, normalize=True):
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

    def fake_heatmap(image, model, processor, scale, device, normalize=True):
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
