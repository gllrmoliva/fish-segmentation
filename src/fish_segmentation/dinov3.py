from dataclasses import dataclass
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


PATCH_SIZE = 16
DEFAULT_MAX_PATCHES = 4096
DEFAULT_TILE_OVERLAP = 0.25


@dataclass(frozen=True)
class AnomalyOptions:
    """Score and mask knobs shared by the end-to-end detectors.

    Defaults are the A/B-validated pipeline: mean-similarity heatmap + hysteresis
    mask. The original adaptive mask is still reachable with
    ``mask_method="adaptive"``, and ``score_mode="knn"`` / local contrast are
    opt-in alternatives (see ``notebooks/08_dinov3_heatmap_ab.ipynb``).
    """

    score_mode: Literal["mean", "knn"] = "mean"
    knn_k: int = 5
    local_contrast_strength: float = 0.0
    local_contrast_sigma_pct: float = 0.05
    mask_method: Literal["adaptive", "hysteresis"] = "hysteresis"
    low_percentile: float = 98.0
    closing_kernel_size: int = 0


class RawPoint(NamedTuple):
    """One salient region emitted by a DINO view, in cropped-frame pixels.

    ``x``/``y``/``radius``/``bbox`` are absolute in the letterbox-cropped frame
    (view origin already added); ``level`` is the recursion depth of the view
    that emitted it. ``bbox`` is the tight component box, so it can be used as a
    SAM3 box prompt once normalized. The metadata defaults keep hand-built test
    points concise.
    """

    x: int
    y: int
    radius: float
    level: int
    area: int = 0
    solidity: float = 0.0
    score_mean: float = 0.0
    score_p95: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None


def letterbox_box(
    image: Image.Image, threshold: float = 0.04
) -> Tuple[int, int, int, int]:
    """Returns the (x0, y0, x1, y1) content box of a letterboxed image.

    Falls back to the full image when every pixel is at or below the threshold.
    Exposed separately from ``remove_letterbox`` so callers that keep the
    original frame (SAM3 point prompts) can map cropped coordinates back to the
    video frame.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    if gray.max() > 1.0:
        gray /= 255.0

    non_black = np.where(gray > threshold)
    if non_black[0].size == 0:
        return (0, 0, image.size[0], image.size[1])
    y_min, y_max = int(non_black[0].min()), int(non_black[0].max())
    x_min, x_max = int(non_black[1].min()), int(non_black[1].max())
    return (x_min, y_min, x_max + 1, y_max + 1)


def remove_letterbox(image: Image.Image, threshold: float = 0.04) -> Image.Image:
    """Detects and crops uniform black letterbox borders from a PIL image."""
    return image.crop(letterbox_box(image, threshold))


def _scaled_patch_shape(
    image_size: Tuple[int, int], scale: float, patch_size: int = PATCH_SIZE
) -> Tuple[int, int]:
    """Returns the patch grid after scaling an image to patch-aligned dimensions."""
    if scale <= 0:
        raise ValueError("scale must be greater than zero")
    if patch_size <= 0:
        raise ValueError("patch_size must be greater than zero")

    orig_w, orig_h = image_size
    scaled_w = max(patch_size, int(round(orig_w * scale / patch_size)) * patch_size)
    scaled_h = max(patch_size, int(round(orig_h * scale / patch_size)) * patch_size)
    return scaled_h // patch_size, scaled_w // patch_size


def prepare_scaled_tensor(
    image: Image.Image,
    processor: AutoImageProcessor,
    scale: float = 4.0,
    patch_size: int = PATCH_SIZE,
    device: str = "cuda",
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Rescales an image to a patch-aligned resolution and prepares pixel tensors."""
    orig_w, orig_h = image.size
    h_patches, w_patches = _scaled_patch_shape(
        image.size, scale=scale, patch_size=patch_size
    )
    scaled_w = w_patches * patch_size
    scaled_h = h_patches * patch_size

    scaled_img = image.resize((scaled_w, scaled_h), Image.Resampling.BICUBIC)
    inputs = processor(images=scaled_img, do_resize=False, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    return pixel_values, (h_patches, w_patches), (orig_w, orig_h)


def load_backbone(
    model_id: str, device: str = "cuda"
) -> Tuple[AutoModel, AutoImageProcessor]:
    """Loads the ViT model and corresponding image processor."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    return model, processor


def extract_patch_tokens(
    model: AutoModel, pixel_values: torch.Tensor, num_patches: int
) -> torch.Tensor:
    """Runs a forward pass and extracts L2-normalized spatial patch tokens."""
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    tokens = outputs.last_hidden_state[0, -num_patches:, :]
    return F.normalize(tokens, p=2, dim=-1)


def _local_contrast(
    scores: np.ndarray, strength: float, sigma_pct: float
) -> np.ndarray:
    """High-passes a score map to flatten broad halos around peaks.

    The single-pass path applies it to the patch grid before border suppression
    and upsampling (the border fill cannot blur into a bright rim and sigma stays
    in patch units); the tiled path applies it once to the stitched map.
    """
    if strength <= 0:
        return scores
    if sigma_pct <= 0:
        raise ValueError("local_contrast_sigma_pct must be greater than zero")

    sigma = max(1.0, sigma_pct * max(scores.shape))
    background = cv2.GaussianBlur(scores.astype(np.float32), (0, 0), sigma)
    return scores - strength * background


def compute_anomaly_heatmap(
    tokens: torch.Tensor,
    grid_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    max_sample_pool: int = 1500,
    border_margin_pct: float = 0.05,
    normalize: bool = True,
    score_mode: Literal["mean", "knn"] = "mean",
    knn_k: int = 5,
    local_contrast_strength: float = 0.0,
    local_contrast_sigma_pct: float = 0.05,
) -> np.ndarray:
    """
    Calculates cosine-distance anomaly scores, suppresses border tokens,
    and bicubic-upsamples back to the target pixel resolution.

    Args:
        normalize: min-max scale the result to [0, 1]. Tiling passes False so
            raw scores from different tiles stay comparable; the stitched map is
            normalized once at the end.
        score_mode: 'mean' uses 1 - mean similarity against the reference pool.
            'knn' uses 1 - mean of the top-k similarities instead: water repeats
            everywhere so it keeps close neighbours, while object patches do not,
            which fills the object interior instead of ringing around it.
        knn_k: nearest references averaged per patch in 'knn' mode.
        local_contrast_strength: >0 subtracts that multiple of a Gaussian-blurred
            copy of the grid scores (unsharp/high-pass), flattening the warm halo
            that surrounds the hottest patches. 0 disables it.
        local_contrast_sigma_pct: blur sigma as a fraction of the longer grid
            side; only used when local_contrast_strength > 0. A small sigma
            (~0.05) removes the mid-scale halo and is meant for the coarse view /
            heatmap inspection; a large sigma (~0.25) only removes the
            very-low-frequency positional fog and survives the zoom recursion
            (objects bigger than the blur are not hollowed into rings).
    """
    if score_mode not in ("mean", "knn"):
        raise ValueError("score_mode must be 'mean' or 'knn'")
    if knn_k <= 0:
        raise ValueError("knn_k must be greater than zero")
    if local_contrast_strength < 0:
        raise ValueError("local_contrast_strength must be non-negative")

    h_patches, w_patches = grid_shape
    num_patches = h_patches * w_patches
    orig_w, orig_h = target_shape

    # Contrast against sampled reference pool to avoid O(N^2) memory bottlenecks
    if num_patches > (max_sample_pool * 2):
        # Fixed seed: the same reference pattern is reused every call, so tiled
        # bias subtraction and repeated runs stay reproducible.
        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(0)
        sample_idx = torch.randperm(
            num_patches, generator=generator, device=tokens.device
        )[:max_sample_pool]
        similarity_matrix = torch.matmul(tokens, tokens[sample_idx].T)
    else:
        similarity_matrix = torch.matmul(tokens, tokens.T)

    if score_mode == "knn":
        k = min(knn_k, similarity_matrix.shape[1])
        anomaly_scores = (1.0 - similarity_matrix.topk(k, dim=1).values.mean(dim=1))
    else:
        anomaly_scores = 1.0 - similarity_matrix.mean(dim=1)
    anomaly_scores = anomaly_scores.reshape(h_patches, w_patches)

    grid_scores = _local_contrast(
        anomaly_scores.detach().cpu().numpy(),
        strength=local_contrast_strength,
        sigma_pct=local_contrast_sigma_pct,
    )

    # ViT border artifact suppression
    border_y = max(1, int(h_patches * border_margin_pct))
    border_x = max(1, int(w_patches * border_margin_pct))
    clean_scores = grid_scores.copy()
    fill_val = clean_scores.min()

    clean_scores[:border_y, :] = fill_val
    clean_scores[-border_y:, :] = fill_val
    clean_scores[:, :border_x] = fill_val
    clean_scores[:, -border_x:] = fill_val

    # Upsample to target spatial dimensions
    anomaly_tensor = torch.as_tensor(clean_scores, dtype=torch.float32).unsqueeze(
        0
    ).unsqueeze(0)
    upsampled = (
        F.interpolate(
            anomaly_tensor, size=(orig_h, orig_w), mode="bicubic", align_corners=False
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    if not normalize:
        return upsampled

    # Min-max normalization
    norm_score = (upsampled - upsampled.min()) / (
        upsampled.max() - upsampled.min() + 1e-8
    )
    return norm_score


def _patch_count(image_size: Tuple[int, int], scale: float) -> int:
    h_patches, w_patches = _scaled_patch_shape(image_size, scale)
    return h_patches * w_patches


def _view_scale(image: Image.Image, long_side: int, resolution_scale: float) -> float:
    """Combines the coarse-view cap with a relative resolution multiplier."""
    if long_side <= 0:
        raise ValueError("long_side must be greater than zero")
    if resolution_scale <= 0:
        raise ValueError("resolution_scale must be greater than zero")

    return min(1.0, long_side / max(image.size)) * resolution_scale


def _tile_starts(length: int, tile_size: int, overlap: float) -> List[int]:
    if not 0 <= overlap < 1:
        raise ValueError("tile overlap must be in the range [0, 1)")
    if tile_size >= length:
        return [0]

    step = max(1, int(round(tile_size * (1 - overlap))))
    starts = list(range(0, length - tile_size + 1, step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _tile_size_for_scale(scale: float, max_patches: int) -> int:
    """Chooses a square native-image tile whose scaled grid fits the patch budget."""
    if max_patches <= 0:
        raise ValueError("max_patches must be greater than zero")

    scaled_side = max(PATCH_SIZE, int(np.sqrt(max_patches)) * PATCH_SIZE)
    tile_size = max(
        PATCH_SIZE,
        int(scaled_side / scale) // PATCH_SIZE * PATCH_SIZE,
    )
    while (
        tile_size > PATCH_SIZE
        and _patch_count((tile_size, tile_size), scale) > max_patches
    ):
        tile_size -= PATCH_SIZE
    return tile_size


def _run_dino_heatmap(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    scale: float,
    device: str,
    normalize: bool = True,
    score_mode: Literal["mean", "knn"] = "mean",
    knn_k: int = 5,
    local_contrast_strength: float = 0.0,
    local_contrast_sigma_pct: float = 0.05,
) -> np.ndarray:
    pixel_values, (h_patches, w_patches), orig_shape = prepare_scaled_tensor(
        image, processor, scale=scale, device=device
    )
    tokens = extract_patch_tokens(model, pixel_values, num_patches=h_patches * w_patches)
    return compute_anomaly_heatmap(
        tokens,
        grid_shape=(h_patches, w_patches),
        target_shape=orig_shape,
        normalize=normalize,
        score_mode=score_mode,
        knn_k=knn_k,
        local_contrast_strength=local_contrast_strength,
        local_contrast_sigma_pct=local_contrast_sigma_pct,
    )


def _tile_window(
    shape: Tuple[int, int],
    x0: int,
    y0: int,
    width: int,
    height: int,
    overlap_px: int,
) -> np.ndarray:
    """Cosine feather weights: 1 inside a tile, tapering over the overlap band.

    Sides that coincide with the image border are not tapered, so edge pixels
    keep full weight. Adjacent tiles' tapers sum to 1 across the overlap band.
    """
    th, tw = shape
    wx = np.ones(tw, dtype=np.float32)
    wy = np.ones(th, dtype=np.float32)
    if overlap_px > 0:
        m = min(overlap_px, tw // 2, th // 2)
        if m > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)).astype(np.float32)
            if x0 > 0:
                wx[:m] = ramp
            if x0 + tw < width:
                wx[-m:] = ramp[::-1]
            if y0 > 0:
                wy[:m] = ramp
            if y0 + th < height:
                wy[-m:] = ramp[::-1]
    return np.outer(wy, wx)


def _tiled_dino_heatmap(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    scale: float,
    device: str,
    max_patches: int,
    tile_overlap: float = DEFAULT_TILE_OVERLAP,
    score_mode: Literal["mean", "knn"] = "mean",
    knn_k: int = 5,
    local_contrast_strength: float = 0.0,
    local_contrast_sigma_pct: float = 0.05,
) -> np.ndarray:
    """Computes a high-resolution heatmap from overlapping tiles.

    Each tile is an independent image for the ViT, so it carries a smooth
    positional bias (bright borders, dark center) that would otherwise repeat on
    the tile lattice as a grid. The stitched map uses the cross-tile phase mean
    -- the average raw score at the same position inside every tile -- as a
    self-calibrated estimate of that repeating component and subtracts it per
    tile, with no extra model passes. Tiles are then blended with a cosine
    feather (so overlap-count steps cannot show as seams) and min-max normalized
    once at the end. Tiles whose shape appears only once (degenerate views) keep
    their raw scores.

    Local contrast is applied to the stitched map, never per tile: high-passing
    each tile amplifies its own water texture and destroys cross-tile
    comparability, which surface as scattered specks after stitching.
    """
    width, height = image.size
    tile_size = _tile_size_for_scale(scale, max_patches)
    step = max(1, int(round(tile_size * (1 - tile_overlap))))
    overlap_px = max(0, tile_size - step)
    x_starts = _tile_starts(width, tile_size, tile_overlap)
    y_starts = _tile_starts(height, tile_size, tile_overlap)

    tiles: list[tuple[int, int, np.ndarray]] = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1, y1 = min(width, x0 + tile_size), min(height, y0 + tile_size)
            tile = image.crop((x0, y0, x1, y1))
            try:
                tile_score = _run_dino_heatmap(
                    tile,
                    model,
                    processor,
                    scale=scale,
                    device=device,
                    normalize=False,
                    score_mode=score_mode,
                    knn_k=knn_k,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.device(device).type != "cuda" or max_patches <= 1:
                    raise
                torch.cuda.empty_cache()
                return _tiled_dino_heatmap(
                    image,
                    model,
                    processor,
                    scale=scale,
                    device=device,
                    max_patches=max(1, max_patches // 2),
                    tile_overlap=tile_overlap,
                    score_mode=score_mode,
                    knn_k=knn_k,
                    local_contrast_strength=local_contrast_strength,
                    local_contrast_sigma_pct=local_contrast_sigma_pct,
                )
            tiles.append((x0, y0, tile_score))

    # Repeating (lattice) component: mean raw score per within-tile position.
    by_shape: dict[tuple[int, int], list[np.ndarray]] = {}
    for _, _, raw in tiles:
        by_shape.setdefault(raw.shape, []).append(raw)
    phase_mean = {
        shape: np.mean(np.stack(maps, axis=0), axis=0)
        for shape, maps in by_shape.items()
        if len(maps) > 1
    }

    score_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for x0, y0, raw in tiles:
        corrected = raw - phase_mean.get(raw.shape, 0.0)
        weight = _tile_window(raw.shape, x0, y0, width, height, overlap_px)
        th, tw = raw.shape
        score_sum[y0:y0 + th, x0:x0 + tw] += corrected * weight
        weight_sum[y0:y0 + th, x0:x0 + tw] += weight

    stitched = score_sum / np.maximum(weight_sum, 1e-6)
    stitched = _local_contrast(
        stitched,
        strength=local_contrast_strength,
        sigma_pct=local_contrast_sigma_pct,
    )
    span = float(stitched.max() - stitched.min())
    if span < 1e-6:
        return np.zeros_like(stitched)
    return (stitched - stitched.min()) / span


def _compute_dino_heatmap(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    scale: float,
    device: str,
    max_patches: int = DEFAULT_MAX_PATCHES,
    tile_overlap: float = DEFAULT_TILE_OVERLAP,
    score_mode: Literal["mean", "knn"] = "mean",
    knn_k: int = 5,
    local_contrast_strength: float = 0.0,
    local_contrast_sigma_pct: float = 0.05,
) -> np.ndarray:
    """Runs one DINO view directly or in tiles when its patch grid is too large."""
    if max_patches <= 0:
        raise ValueError("max_patches must be greater than zero")

    heatmap_kwargs = {
        "score_mode": score_mode,
        "knn_k": knn_k,
        "local_contrast_strength": local_contrast_strength,
        "local_contrast_sigma_pct": local_contrast_sigma_pct,
    }
    patch_count = _patch_count(image.size, scale)
    if patch_count > max_patches:
        return _tiled_dino_heatmap(
            image,
            model,
            processor,
            scale=scale,
            device=device,
            max_patches=max_patches,
            tile_overlap=tile_overlap,
            **heatmap_kwargs,
        )

    try:
        return _run_dino_heatmap(
            image, model, processor, scale=scale, device=device, **heatmap_kwargs
        )
    except torch.cuda.OutOfMemoryError:
        if torch.device(device).type != "cuda":
            raise
        torch.cuda.empty_cache()
        return _tiled_dino_heatmap(
            image,
            model,
            processor,
            scale=scale,
            device=device,
            max_patches=max(1, max_patches // 2),
            tile_overlap=tile_overlap,
            **heatmap_kwargs,
        )


def _grow_seeds(seeds: np.ndarray, low_mask: np.ndarray) -> np.ndarray:
    """Hysteresis growth: keeps low-threshold components that contain a seed.

    Equivalent to morphological reconstruction of ``low_mask`` from ``seeds``,
    done with connected components so long thin bridges connect exactly.
    """
    num_labels, labels = cv2.connectedComponents(
        low_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(low_mask, dtype=np.uint8)

    seed_labels = np.unique(labels[seeds > 0])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return np.zeros_like(low_mask, dtype=np.uint8)
    return np.isin(labels, seed_labels).astype(np.uint8)


def segment_anomalies(
    norm_score: np.ndarray,
    percentile_threshold: float = 99.5,
    adaptive_block_size: int = 21,
    morph_kernel_size: int = 3,
    method: Literal["adaptive", "hysteresis"] = "hysteresis",
    low_percentile: float = 98.0,
    closing_kernel_size: int = 0,
) -> np.ndarray:
    """Builds the anomaly mask with global percentile filtering and morphology.

    Args:
        method: 'hysteresis' (default) seeds with the top percentile and grows
            every seed through the connected region above ``low_percentile`` —
            one connected component per object instead of a fragmented core,
            without flooding the whole warm halo. 'adaptive' is the original
            top-percentile mask ANDed with a local adaptive-threshold mask; its
            local core can be empty on large frames (smooth peaks never beat
            their own 21 px window mean), so it misses whole frames.
        low_percentile: lower growth threshold for 'hysteresis'
            (0 < low_percentile <= percentile_threshold).
        closing_kernel_size: optional morphological closing (ellipse, px) before
            the final opening; bridges small gaps between fragments. 0 disables.
    """
    if method not in ("adaptive", "hysteresis"):
        raise ValueError("method must be 'adaptive' or 'hysteresis'")
    if closing_kernel_size < 0:
        raise ValueError("closing_kernel_size must be non-negative")

    if method == "hysteresis":
        if not 0 < low_percentile <= percentile_threshold <= 100:
            raise ValueError(
                "percentiles must satisfy 0 < low_percentile <= percentile_threshold <= 100"
            )
        high_val = np.percentile(norm_score, percentile_threshold)
        low_val = np.percentile(norm_score, low_percentile)
        seeds = (norm_score >= high_val).astype(np.uint8)
        low_mask = (norm_score >= low_val).astype(np.uint8)
        fine_mask = _grow_seeds(seeds, low_mask) * 255

        if closing_kernel_size > 1:
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
            )
            fine_mask = cv2.morphologyEx(fine_mask, cv2.MORPH_CLOSE, close_kernel)
    else:
        # Top-tail percentile mask
        threshold_val = np.percentile(norm_score, percentile_threshold)
        binary_mask = (norm_score >= threshold_val).astype(np.uint8) * 255

        # Local adaptive contrast refinement
        roi = (norm_score * 255).astype(np.uint8)
        core_mask = cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size,
            -2,
        )
        fine_mask = cv2.bitwise_and(binary_mask, core_mask)

    # Morphological noise removal
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    return cv2.morphologyEx(fine_mask, cv2.MORPH_OPEN, kernel)


def extract_salient_regions(
    binary_mask: np.ndarray,
    method: Literal["otsu", "iqr"] = "otsu",
    iqr_k: float = 1.5,
    min_absolute_pixels: int = 20,
    score_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Filtra componentes pequeños... — translated:
    Filters small components from a binary mask and extracts guaranteed interior
    points (inaccessibility poles) suitable for complex/torus geometries.

    Args:
        binary_mask: uint8 binary mask (0/255), e.g. the output of segment_anomalies().
        method: 'otsu' — Otsu over log(1 + area); best when background noise and
                clear regions of interest coexist.
                'iqr' — Q3 + k*IQR outlier rule; best when the mask is almost only
                noise with a few large isolated blobs.
        iqr_k: IQR multiplier (typically 1.5 for standard outliers, 0 for >= Q3).
        min_absolute_pixels: hard pre-filter cutoff for insignificant 1-N pixel artifacts.
        score_map: optional anomaly heatmap aligned with ``binary_mask``; when
            given, per-region ``score_mean``/``score_p95`` are added.

    Returns:
        filtered_mask: clean mask containing only the selected regions.
        regions_info: per-region metadata dicts:
            - 'label': connected-component ID.
            - 'area': area in pixels.
            - 'center': (x, y) coordinates guaranteed to be inside the region.
            - 'max_inscribed_radius': maximum Euclidean distance to the boundary.
            - 'bbox': tight (x0, y0, x1, y1) component box.
            - 'solidity': ``area / (pi * r^2)``; ~1 for a disk, ~1.27 for a
              square, larger for elongated or hollow components.
            - 'score_mean', 'score_p95': anomaly score stats inside the
              component (only when ``score_map`` is given).
    """
    mask = (binary_mask > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(mask), []

    # Extraer áreas omitiendo el fondo (índice 0)
    all_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)

    # Pre-filter microscopic noise
    valid_indices = np.where(all_areas >= min_absolute_pixels)[0]
    if len(valid_indices) == 0:
        return np.zeros_like(mask), []

    candidate_areas = all_areas[valid_indices]

    # 1. Statistical area threshold
    if method == "otsu" and len(candidate_areas) > 2:
        log_areas = np.log1p(candidate_areas)
        min_log, max_log = log_areas.min(), log_areas.max()

        if max_log > min_log:
            norm_log = np.uint8(255 * (log_areas - min_log) / (max_log - min_log))
            thresh_val, _ = cv2.threshold(
                norm_log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            threshold_log = min_log + (thresh_val / 255.0) * (max_log - min_log)
            area_threshold = np.expm1(threshold_log)
        else:
            area_threshold = candidate_areas.min()

    elif method == "iqr":
        q1 = np.percentile(candidate_areas, 25)
        q3 = np.percentile(candidate_areas, 75)
        iqr = q3 - q1
        area_threshold = q3 + (iqr_k * iqr) if iqr > 0 else np.median(candidate_areas)

    else:
        # Defensive fallback when there are too few regions for Otsu
        area_threshold = np.median(candidate_areas)

    # 2. Extract regions and inaccessibility poles
    filtered_mask = np.zeros_like(mask)
    regions_info = []

    for idx in valid_indices:
        label = idx + 1  # background offset
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= area_threshold:
            comp_mask = np.uint8(labels == label)
            filtered_mask[comp_mask > 0] = 255

            # Euclidean distance transform
            dist_map = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
            radius = float(max_val)

            bx, by, bw, bh = (int(v) for v in stats[label, :4])
            info = {
                "label": label,
                "area": area,
                "center": max_loc,  # (column_x, row_y) inside the area
                "max_inscribed_radius": radius,
                "bbox": (bx, by, bx + bw, by + bh),
                "solidity": area / (np.pi * radius**2) if radius > 0 else 0.0,
            }
            if score_map is not None:
                values = score_map[comp_mask > 0]
                info["score_mean"] = float(values.mean())
                info["score_p95"] = float(np.percentile(values, 95))
            regions_info.append(info)

    return filtered_mask, regions_info


def run_dino_detector(
    raw_image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    resolution_scale: float = 4.0,
    percentile_threshold: float = 99.5,
    visualize: bool = True,
    device: Optional[str] = None,
    max_patches: int = DEFAULT_MAX_PATCHES,
    options: AnomalyOptions = AnomalyOptions(),
) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    """
    End-to-end execution pipeline for ViT-based anomaly segmentation.
    Returns: (final_mask, norm_score, cropped_image)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cropped_img = remove_letterbox(raw_image)
    norm_score = _compute_dino_heatmap(
        cropped_img,
        model,
        processor,
        scale=resolution_scale,
        device=device,
        max_patches=max_patches,
        score_mode=options.score_mode,
        knn_k=options.knn_k,
        local_contrast_strength=options.local_contrast_strength,
        local_contrast_sigma_pct=options.local_contrast_sigma_pct,
    )
    final_mask = segment_anomalies(
        norm_score,
        percentile_threshold=percentile_threshold,
        method=options.mask_method,
        low_percentile=options.low_percentile,
        closing_kernel_size=options.closing_kernel_size,
    )

    return final_mask, norm_score, cropped_img


def _dino_pass(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    long_side: int,
    device: str,
    percentile_threshold: float = 99.5,
    resolution_scale: float = 1.0,
    max_patches: int = DEFAULT_MAX_PATCHES,
    options: AnomalyOptions = AnomalyOptions(),
    trace: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs DINO on one image view.

    Returns the binary anomaly mask and the normalized heatmap, both at native
    view size. When ``trace`` is given, it is filled in-place with the view
    ``scale`` and the raw normalized anomaly heatmap (``norm_score``), which the
    zoom trace uses to render layer-by-layer diagnostics.
    """
    scale = _view_scale(image, long_side, resolution_scale)
    norm_score = _compute_dino_heatmap(
        image,
        model,
        processor,
        scale=scale,
        device=device,
        max_patches=max_patches,
        score_mode=options.score_mode,
        knn_k=options.knn_k,
        local_contrast_strength=options.local_contrast_strength,
        local_contrast_sigma_pct=options.local_contrast_sigma_pct,
    )
    if trace is not None:
        trace["scale"] = scale
        trace["norm_score"] = norm_score
    mask = segment_anomalies(
        norm_score,
        percentile_threshold=percentile_threshold,
        method=options.mask_method,
        low_percentile=options.low_percentile,
        closing_kernel_size=options.closing_kernel_size,
    )
    return mask, norm_score


def run_dino_view(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    long_side: int = 1024,
    resolution_scale: float = 1.0,
    percentile_threshold: float = 99.5,
    device: Optional[str] = None,
    max_patches: int = DEFAULT_MAX_PATCHES,
    options: AnomalyOptions = AnomalyOptions(),
) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    """Runs one zoom-compatible DINO view and returns its mask and heatmap."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cropped_img = remove_letterbox(image)
    scale = _view_scale(cropped_img, long_side, resolution_scale)
    norm_score = _compute_dino_heatmap(
        cropped_img,
        model,
        processor,
        scale=scale,
        device=device,
        max_patches=max_patches,
        score_mode=options.score_mode,
        knn_k=options.knn_k,
        local_contrast_strength=options.local_contrast_strength,
        local_contrast_sigma_pct=options.local_contrast_sigma_pct,
    )
    final_mask = segment_anomalies(
        norm_score,
        percentile_threshold=percentile_threshold,
        method=options.mask_method,
        low_percentile=options.low_percentile,
        closing_kernel_size=options.closing_kernel_size,
    )
    return final_mask, norm_score, cropped_img


def _boxes_within(a: list, b: list, gap: float) -> bool:
    """True when two boxes are closer than ``gap`` px (gap=0 means overlap)."""
    return (
        a[0] - gap < b[2]
        and a[2] + gap > b[0]
        and a[1] - gap < b[3]
        and a[3] + gap > b[1]
    )


def _interest_zones(
    mask: np.ndarray,
    margin: float = 0.3,
    min_area: int = 30,
    max_zone_fraction: float = 0.8,
    merge_gap: float = 0.0,
    min_zone_fraction: float = 0.0,
) -> list[tuple[int, int, int, int]]:
    """Binary mask -> expanded, merged bounding boxes of dense regions (native coords).

    Small components are dropped before expansion so noise specks never form zones.
    Boxes closer than ``merge_gap`` (fraction of the view's long side) are merged
    into one zone, so a single DINO pass sees several nearby objects with shared
    context; ``merge_gap=0`` merges only overlapping boxes. Every zone is then
    grown, centered and clamped to the image until both sides reach at least
    ``min_zone_fraction`` of the long side, and zones covering more than
    ``max_zone_fraction`` of the view are dropped (zooming gains nothing there).
    """
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    if not 0 <= min_zone_fraction <= 1:
        raise ValueError("min_zone_fraction must be in the range [0, 1]")

    h, w = mask.shape
    long_side = max(w, h)
    gap_px = merge_gap * long_side
    min_side_px = min_zone_fraction * long_side

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw = []
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        mx, my = int(bw * margin), int(bh * margin)
        raw.append(
            [
                max(0, x - mx),
                max(0, y - my),
                min(w, x + bw + mx),
                min(h, y + bh + my),
            ]
        )

    # Union-find merge of boxes closer than gap_px (order-independent)
    parent = list(range(len(raw)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(raw)):
        for j in range(i + 1, len(raw)):
            if _boxes_within(raw[i], raw[j], gap_px):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    zones: list[list[int]] = []
    zone_of_root: dict[int, list[int]] = {}
    for i, box in enumerate(raw):
        root = find(i)
        if root not in zone_of_root:
            zone_of_root[root] = list(box)
            zones.append(zone_of_root[root])
        else:
            zone = zone_of_root[root]
            zone[0], zone[1] = min(zone[0], box[0]), min(zone[1], box[1])
            zone[2], zone[3] = max(zone[2], box[2]), max(zone[3], box[3])

    # Enforce a minimum context area, centered on each zone and clamped to the view
    for zone in zones:
        if min_side_px <= 0:
            continue
        target = max(min_side_px, zone[2] - zone[0], zone[3] - zone[1])
        cx = (zone[0] + zone[2]) / 2
        cy = (zone[1] + zone[3]) / 2
        zone[0], zone[2] = int(round(cx - target / 2)), int(round(cx + target / 2))
        zone[1], zone[3] = int(round(cy - target / 2)), int(round(cy + target / 2))
        zone[0], zone[1] = max(0, zone[0]), max(0, zone[1])
        zone[2], zone[3] = min(w, zone[2]), min(h, zone[3])

    return [
        (x0, y0, x1, y1)
        for x0, y0, x1, y1 in zones
        if (x1 - x0) * (y1 - y0) <= max_zone_fraction * w * h
    ]


def _circles_overlap(a: RawPoint, b: RawPoint) -> bool:
    """True when the two inscribed circles overlap."""
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 < (a.radius + b.radius) ** 2


def _detection_matches(a: RawPoint, b: RawPoint) -> bool:
    """True when two raw points from different levels likely belong to one object.

    Circle overlap alone misses wide or hollow coarse regions whose inscribed
    center sits far from a deep detection near the region edge, so containment
    of one center inside the other's region box also counts.
    """
    if _circles_overlap(a, b):
        return True
    for point, box in ((a, b.bbox), (b, a.bbox)):
        if box is None:
            continue
        if box[0] <= point.x <= box[2] and box[1] <= point.y <= box[3]:
            return True
    return False


def _cluster_level(points: list[RawPoint]) -> list[list[RawPoint]]:
    """Transitive clustering of same-level points (conservative circle test).

    Two points join when one center falls inside the other's inscribed circle
    (``dist < max(r_i, r_j)``): strict enough to keep two close animals apart,
    loose enough to dedupe the same region seen by overlapping views.  Uses
    union-find, so the result does not depend on the input order.
    """
    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a, b = points[i], points[j]
            if (a.x - b.x) ** 2 + (a.y - b.y) ** 2 < max(a.radius, b.radius) ** 2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[RawPoint]] = {}
    for i, point in enumerate(points):
        groups.setdefault(find(i), []).append(point)
    return list(groups.values())


def _merge_points(
    points: list[RawPoint],
    keep_absorbed: bool = False,
) -> list[dict]:
    """Deepest-wins merge of raw region points across zoom levels.

    Same-level points are clustered transitively first. Clusters are then
    resolved from the deepest level to the coarsest: a representative that
    matches an already kept (deeper) cluster — inscribed circles overlap or one
    center falls inside the other's region box — is *absorbed* into it: it only
    adds confirmation (``n_levels``/``n_points``) and is not emitted — so the
    emitted center and radius always belong to a real region of the deepest
    level and never to an average across levels. A shallow representative that
    matches no deeper cluster is kept as its own detection. With
    ``keep_absorbed=True`` the absorbed representative is emitted as well, for
    A/B comparisons.

    Deterministic: clustering is order-independent and each cluster's
    representative is its largest inscribed circle (ties: largest area, then
    topmost-leftmost coordinates).

    Returns one dict per detection with pixel coords (``x``, ``y``, ``radius``,
    ``bbox``) plus ``level``, ``area``, ``solidity``, ``score_mean``,
    ``score_p95``, ``n_levels``, ``levels`` and ``n_points``.
    """
    if not points:
        return []

    def representative(members: list[RawPoint]) -> RawPoint:
        return max(members, key=lambda p: (p.radius, p.area, -p.y, -p.x))

    def cluster_record(rep: RawPoint, members: list[RawPoint]) -> dict:
        return {"rep": rep, "levels": {rep.level}, "n_points": len(members)}

    by_level: dict[int, list[RawPoint]] = {}
    for point in points:
        by_level.setdefault(point.level, []).append(point)

    kept: list[dict] = []
    for level in sorted(by_level, reverse=True):
        for members in _cluster_level(by_level[level]):
            rep = representative(members)
            hits = [k for k in kept if _detection_matches(rep, k["rep"])]
            if hits:
                for k in hits:
                    k["levels"].add(level)
                    k["n_points"] += len(members)
                if keep_absorbed:
                    kept.append(cluster_record(rep, members))
            else:
                kept.append(cluster_record(rep, members))

    detections = []
    for k in kept:
        rep: RawPoint = k["rep"]
        detections.append(
            {
                "x": rep.x,
                "y": rep.y,
                "radius": rep.radius,
                "level": rep.level,
                "area": rep.area,
                "solidity": rep.solidity,
                "score_mean": rep.score_mean,
                "score_p95": rep.score_p95,
                "bbox": rep.bbox,
                "n_levels": len(k["levels"]),
                "levels": tuple(sorted(k["levels"])),
                "n_points": k["n_points"],
            }
        )
    detections.sort(key=lambda d: (-d["level"], d["y"], d["x"]))
    return detections


def run_zoom_detector(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    long_side: int = 1024,
    margin: float = 0.3,
    min_object_pixels: int = 30,
    percentile_threshold: float = 99.5,
    max_levels: int = 4,
    device: Optional[str] = None,
    resolution_scale: float = 1.0,
    max_patches: int = DEFAULT_MAX_PATCHES,
    merge_gap: float = 0.0,
    min_zone_fraction: float = 0.0,
    min_confirmations: int = 1,
    keep_absorbed: bool = False,
    options: AnomalyOptions = AnomalyOptions(),
    trace: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Coarse-to-fine zoom detection with an optional relative resolution multiplier.
    DINO runs on a <=long_side view, multiplied by resolution_scale, then crops
    interest zones from the ORIGINAL pixels and re-detects until the view is
    native and <=long_side. Oversized scaled views are processed in overlapping
    tiles. Points are merged across levels with the deepest-wins rule and
    returned normalized to the ORIGINAL frame (letterbox bars included), so the
    coordinates can feed SAM3 point prompts directly.

    ``merge_gap`` and ``min_zone_fraction`` (both fractions of the view's long
    side) control the context of each zoom crop: boxes closer than ``merge_gap``
    are fused into one zone so a single DINO pass sees several nearby objects,
    and every zone is grown to at least ``min_zone_fraction`` of the long side.
    The defaults (0) keep the original per-component crops.

    ``min_confirmations`` keeps only detections that match regions from at least
    that many recursion depths (1 = keep everything).
    ``keep_absorbed`` emits coarse representatives that were absorbed by deeper
    detections as well, for A/B comparisons of the merge rule.

    ``options`` bundles the score mode (mean/knn + local contrast) and the mask
    method (adaptive/hysteresis); see ``AnomalyOptions``.

    When ``trace`` is a list, one record per DINO view is appended in depth-first
    order with keys:
        - 'depth': recursion level.
        - 'origin': (x, y) of the view in the letterbox-cropped original coords.
        - 'size': (w, h) of the view.
        - 'scale': relative resolution used to feed the view.
        - 'feed_native': True when this view stopped the recursion (native size).
        - 'norm_score': normalized anomaly heatmap at view resolution.
        - 'mask': binary anomaly mask at view resolution.
        - 'regions': extract_salient_regions() output (view coords).
        - 'zones': interest-zone boxes this view recursed into (empty when stopped).
        - 'points': raw RawPoint records emitted by this view, pre-merge.

    Returns list of dicts with keys:
        - 'x', 'y': normalized [0,1] center in original-frame coords.
        - 'radius': max inscribed radius of the deepest confirming region,
          normalized by max(W, H) of the original frame.
        - 'level': recursion depth of the deepest region (deeper = finer).
        - 'bbox': normalized tight box of that region (box-prompt candidate).
        - 'area', 'solidity', 'score_mean', 'score_p95': region metadata.
        - 'n_levels', 'levels', 'n_points': merge confirmation counts.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if min_confirmations < 1:
        raise ValueError("min_confirmations must be at least 1")

    x0, y0, x1, y1 = letterbox_box(image)
    W, H = image.size
    view = image.crop((x0, y0, x1, y1))
    points: list[RawPoint] = []

    def recurse(img: Image.Image, ox: int, oy: int, depth: int) -> None:
        w, h = img.size
        feed_native = max(w, h) <= long_side
        entry = None
        if trace is not None:
            entry = {"depth": depth, "origin": (ox, oy), "size": (w, h)}
            trace.append(entry)
        pass_kwargs = {} if entry is None else {"trace": entry}
        mask, norm_score = _dino_pass(
            img, model, processor, long_side, device,
            percentile_threshold=percentile_threshold,
            resolution_scale=resolution_scale,
            max_patches=max_patches,
            options=options,
            **pass_kwargs,
        )
        _, regions = extract_salient_regions(
            mask, min_absolute_pixels=min_object_pixels, score_map=norm_score
        )
        level_points = [
            RawPoint(
                x=ox + reg["center"][0],
                y=oy + reg["center"][1],
                radius=reg["max_inscribed_radius"],
                level=depth,
                area=reg["area"],
                solidity=reg["solidity"],
                score_mean=reg.get("score_mean", 0.0),
                score_p95=reg.get("score_p95", 0.0),
                bbox=(
                    ox + reg["bbox"][0],
                    oy + reg["bbox"][1],
                    ox + reg["bbox"][2],
                    oy + reg["bbox"][3],
                ),
            )
            for reg in regions
        ]
        points.extend(level_points)

        if feed_native or depth >= max_levels:
            zones: list[tuple[int, int, int, int]] = []
        else:
            zones = _interest_zones(
                mask,
                margin=margin,
                merge_gap=merge_gap,
                min_zone_fraction=min_zone_fraction,
            )
        if entry is not None:
            entry["feed_native"] = feed_native
            entry["mask"] = mask
            entry["regions"] = regions
            entry["zones"] = zones
            entry["points"] = level_points

        for zx0, zy0, zx1, zy1 in zones:
            recurse(img.crop((zx0, zy0, zx1, zy1)), ox + zx0, oy + zy0, depth + 1)

    recurse(view, 0, 0, 0)

    detections = []
    for det in _merge_points(points, keep_absorbed=keep_absorbed):
        if det["n_levels"] < min_confirmations:
            continue
        bx0, by0, bx1, by1 = det["bbox"]
        detections.append(
            {
                "x": (det["x"] + x0) / W,
                "y": (det["y"] + y0) / H,
                "radius": det["radius"] / max(W, H),
                "level": det["level"],
                "bbox": (
                    (bx0 + x0) / W,
                    (by0 + y0) / H,
                    (bx1 + x0) / W,
                    (by1 + y0) / H,
                ),
                "area": det["area"],
                "solidity": det["solidity"],
                "score_mean": det["score_mean"],
                "score_p95": det["score_p95"],
                "n_levels": det["n_levels"],
                "levels": det["levels"],
                "n_points": det["n_points"],
            }
        )
    return detections


def plot_detection_results(
    image: Image.Image,
    norm_score: np.ndarray,
    final_mask: np.ndarray,
    overlay_color: Tuple[int, int, int] = (255, 30, 30),
    alpha: float = 0.5,
) -> None:
    """Plots the input, DINO anomaly heatmap, and thresholded mask.

    ``norm_score`` is the raw normalized anomaly signal before thresholding. This
    view helps distinguish a weak model signal from an overly aggressive mask
    or connected-component filter.
    """
    image_array = np.asarray(image.convert("RGB"))
    expected_shape = image_array.shape[:2]
    if norm_score.shape != expected_shape or final_mask.shape != expected_shape:
        raise ValueError(
            "image, norm_score, and final_mask must have matching spatial shapes"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_array)
    axes[0].set_title("Input")
    axes[0].axis("off")

    heatmap = axes[1].imshow(norm_score, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("DINO anomaly heatmap")
    axes[1].axis("off")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    overlay = image_array.copy()
    overlay[final_mask > 0] = overlay_color
    blended = cv2.addWeighted(image_array, 1 - alpha, overlay, alpha, 0)
    axes[2].imshow(blended)
    mask_pixels = int(np.count_nonzero(final_mask))
    axes[2].set_title(f"Thresholded mask ({mask_pixels:,} pixels)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_regions_with_centers(
    image: Image.Image,
    filtered_mask: np.ndarray,
    regions_info: List[Dict],
    show_mask: bool = True,
    overlay_color: Tuple[int, int, int] = (255, 30, 30),
    alpha: float = 0.4,
    point_color: str = "cyan",
    show_radius: bool = True,
) -> None:
    """Plots interior region centers over the image, with optional mask overlay.

    Args:
        show_mask: True overlays a translucent silhouette; False shows the clean
            image with the points only.
        show_radius: draws the maximum inscribed circle around each interior point.
    """
    img_np = np.array(image).copy()

    if show_mask and filtered_mask is not None:
        overlay = img_np.copy()
        overlay[filtered_mask == 255] = overlay_color
        canvas = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)
    else:
        canvas = img_np

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(canvas)

    # Draw points and labels
    for reg in regions_info:
        cx, cy = reg["center"]
        radius = reg["max_inscribed_radius"]

        # Interior point (inaccessibility pole)
        ax.scatter(
            cx, cy, c=point_color, edgecolors="black", s=60, zorder=5, marker="o"
        )

        # Maximum inscribed circle
        if show_radius and radius > 2:
            circle = plt.Circle(
                (cx, cy),
                radius,
                color=point_color,
                fill=False,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )
            ax.add_patch(circle)

        # Component-number label
        ax.annotate(
            f"#{reg['label']}",
            (cx, cy),
            textcoords="offset points",
            xytext=(7, 7),
            color="white",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, lw=0),
        )

    title_suffix = "with mask" if show_mask else "clean image"
    ax.set_title(
        f"Salient Regions ({len(regions_info)} detected) — {title_suffix}"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


ZOOM_LEVEL_COLORS = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def _draw_region_markers(ax, regions_info: List[Dict]) -> None:
    """Draws region interior points and their max inscribed circles on an axis."""
    for reg in regions_info:
        cx, cy = reg["center"]
        radius = reg["max_inscribed_radius"]
        ax.scatter(cx, cy, c="cyan", edgecolors="black", s=60, zorder=5)
        if radius > 2:
            ax.add_patch(
                plt.Circle(
                    (cx, cy),
                    radius,
                    color="cyan",
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                )
            )


def plot_zoom_overview(
    image: Image.Image,
    trace: List[Dict],
    detections: Optional[List[Dict]] = None,
    alpha: float = 0.25,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """Plots the whole zoom tree on one image: every traced view as a box.

    Boxes are colored by recursion depth; final merged ``detections`` (as
    returned by ``run_zoom_detector``) are drawn on top, colored by their
    confirmation level, with their normalized boxes.

    Args:
        image: the original frame passed to ``run_zoom_detector`` (letterbox
            bars are cropped internally).
        trace: the list filled by ``run_zoom_detector(..., trace=trace)``.
        detections: optional merged detections from the same call.
    """
    x0, y0, x1, y1 = letterbox_box(image)
    W, H = image.size
    img_np = np.asarray(image.crop((x0, y0, x1, y1)).convert("RGB"))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img_np)

    depth_colors = {}
    for entry in trace:
        depth = entry["depth"]
        color = ZOOM_LEVEL_COLORS[depth % len(ZOOM_LEVEL_COLORS)]
        depth_colors[depth] = color
        ox, oy = entry["origin"]
        w, h = entry["size"]
        ax.add_patch(
            plt.Rectangle(
                (ox + x0, oy + y0),
                w,
                h,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=1.2,
            )
        )
        ax.annotate(
            f"L{depth}",
            (ox + x0 + 4, oy + y0 + 14),
            color="white",
            fontsize=8,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0),
        )

    for det in detections or []:
        cx, cy = det["x"] * W, det["y"] * H
        radius = det["radius"] * max(W, H)
        color = ZOOM_LEVEL_COLORS[det["level"] % len(ZOOM_LEVEL_COLORS)]
        ax.scatter(cx, cy, c=color, edgecolors="black", s=70, zorder=6)
        if radius > 2:
            ax.add_patch(
                plt.Circle(
                    (cx, cy),
                    radius,
                    color=color,
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.9,
                )
            )
        bbox = det.get("bbox")
        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            ax.add_patch(
                plt.Rectangle(
                    (bx0 * W, by0 * H),
                    (bx1 - bx0) * W,
                    (by1 - by0) * H,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=6,
                )
            )

    if depth_colors:
        handles = [
            plt.Line2D([0], [0], color=c, lw=6, label=f"depth {d}")
            for d, c in sorted(depth_colors.items())
        ]
        ax.legend(handles=handles, loc="upper right", framealpha=0.8)

    n_det = 0 if detections is None else len(detections)
    ax.set_title(f"Zoom tree — {len(trace)} DINO views, {n_det} merged detections")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_zoom_level(
    image: Image.Image,
    entry: Dict,
    overlay_color: Tuple[int, int, int] = (255, 30, 30),
    alpha: float = 0.45,
    figsize: Tuple[float, float] = (24, 6),
) -> None:
    """Plots one traced DINO view as four image panels.

    Panels: the raw view crop fed to DINO, its anomaly heatmap, the thresholded
    mask with region centers/inscribed radii and child zoom zones, and the raw
    points emitted by this view (pre-merge).

    ``image`` must be the original frame passed to ``run_zoom_detector``
    (letterbox bars are cropped internally); the view is reconstructed from
    ``entry['origin']`` and ``entry['size']``, both in cropped-frame coords.
    """
    x0, y0, _, _ = letterbox_box(image)
    ox, oy = entry["origin"]
    w, h = entry["size"]
    view = image.crop((x0 + ox, y0 + oy, x0 + ox + w, y0 + oy + h))
    view_np = np.asarray(view.convert("RGB"))
    mask = entry["mask"]
    regions = entry["regions"]
    zones = entry["zones"]

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(view_np)
    axes[0].set_title(f"layer input — {w}x{h} @ ({ox}, {oy})")

    heat = axes[1].imshow(entry["norm_score"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("anomaly heatmap")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)

    overlay = view_np.copy()
    overlay[mask > 0] = overlay_color
    axes[2].imshow(cv2.addWeighted(view_np, 1 - alpha, overlay, alpha, 0))
    axes[2].set_title(f"mask + {len(regions)} regions + {len(zones)} zones")
    for zx0, zy0, zx1, zy1 in zones:
        axes[2].add_patch(
            plt.Rectangle(
                (zx0, zy0),
                zx1 - zx0,
                zy1 - zy0,
                fill=False,
                edgecolor="cyan",
                linestyle="--",
                linewidth=1.2,
            )
        )
    _draw_region_markers(axes[2], regions)

    axes[3].imshow(view_np)
    axes[3].set_title(f"points emitted ({len(entry['points'])})")
    for point in entry["points"]:
        cx, cy = point.x - ox, point.y - oy
        axes[3].scatter(cx, cy, c="cyan", edgecolors="black", s=60, zorder=5)
        if point.radius > 2:
            axes[3].add_patch(
                plt.Circle(
                    (cx, cy),
                    point.radius,
                    color="cyan",
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                )
            )

    stop = " — native stop" if entry["feed_native"] else ""
    fig.suptitle(
        f"depth {entry['depth']} | scale {entry['scale']:.2f}{stop}"
    )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
