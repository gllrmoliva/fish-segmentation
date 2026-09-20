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


def test_tiled_heatmap_covers_the_native_image(monkeypatch):
    calls = []

    def fake_heatmap(image, model, processor, scale, device):
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
    assert np.allclose(heatmap, 1.0)
    assert all(d._patch_count(size, 2.0) <= 1024 for size in calls)
