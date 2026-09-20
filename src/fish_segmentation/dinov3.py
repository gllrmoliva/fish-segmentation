from typing import Dict, List, Literal, Optional, Tuple

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


def remove_letterbox(image: Image.Image, threshold: float = 0.04) -> Image.Image:
    """Detects and crops uniform black letterbox borders from a PIL image."""
    gray = np.array(image.convert("L"), dtype=np.float32)
    if gray.max() > 1.0:
        gray /= 255.0

    non_black = np.where(gray > threshold)
    if non_black[0].size > 0:
        y_min, y_max = int(non_black[0].min()), int(non_black[0].max())
        x_min, x_max = int(non_black[1].min()), int(non_black[1].max())
        return image.crop((x_min, y_min, x_max + 1, y_max + 1))
    return image


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


def compute_anomaly_heatmap(
    tokens: torch.Tensor,
    grid_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    max_sample_pool: int = 1500,
    border_margin_pct: float = 0.05,
) -> np.ndarray:
    """
    Calculates cosine-distance anomaly scores, suppresses border tokens,
    and bicubic-upsamples back to the target pixel resolution.
    """
    h_patches, w_patches = grid_shape
    num_patches = h_patches * w_patches
    orig_w, orig_h = target_shape

    # Contrast against sampled reference pool to avoid O(N^2) memory bottlenecks
    if num_patches > (max_sample_pool * 2):
        sample_idx = torch.randperm(num_patches, device=tokens.device)[:max_sample_pool]
        similarity_matrix = torch.matmul(tokens, tokens[sample_idx].T)
    else:
        similarity_matrix = torch.matmul(tokens, tokens.T)

    mean_sim = similarity_matrix.mean(dim=1)
    anomaly_scores = (1.0 - mean_sim).reshape(h_patches, w_patches)

    # ViT border artifact suppression
    border_y = max(1, int(h_patches * border_margin_pct))
    border_x = max(1, int(w_patches * border_margin_pct))
    clean_scores = anomaly_scores.clone()
    fill_val = clean_scores.min()

    clean_scores[:border_y, :] = fill_val
    clean_scores[-border_y:, :] = fill_val
    clean_scores[:, :border_x] = fill_val
    clean_scores[:, -border_x:] = fill_val

    # Upsample to target spatial dimensions
    anomaly_tensor = clean_scores.unsqueeze(0).unsqueeze(0)
    upsampled = (
        F.interpolate(
            anomaly_tensor, size=(orig_h, orig_w), mode="bicubic", align_corners=False
        )
        .squeeze()
        .cpu()
        .numpy()
    )

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
) -> np.ndarray:
    pixel_values, (h_patches, w_patches), orig_shape = prepare_scaled_tensor(
        image, processor, scale=scale, device=device
    )
    tokens = extract_patch_tokens(model, pixel_values, num_patches=h_patches * w_patches)
    return compute_anomaly_heatmap(
        tokens, grid_shape=(h_patches, w_patches), target_shape=orig_shape
    )


def _tiled_dino_heatmap(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    scale: float,
    device: str,
    max_patches: int,
    tile_overlap: float = DEFAULT_TILE_OVERLAP,
) -> np.ndarray:
    """Computes a high-resolution heatmap by averaging overlapping tile scores."""
    width, height = image.size
    tile_size = _tile_size_for_scale(scale, max_patches)
    x_starts = _tile_starts(width, tile_size, tile_overlap)
    y_starts = _tile_starts(height, tile_size, tile_overlap)

    score_sum = np.zeros((height, width), dtype=np.float32)
    score_count = np.zeros((height, width), dtype=np.float32)

    for y0 in y_starts:
        for x0 in x_starts:
            x1, y1 = min(width, x0 + tile_size), min(height, y0 + tile_size)
            tile = image.crop((x0, y0, x1, y1))
            try:
                tile_score = _run_dino_heatmap(
                    tile, model, processor, scale=scale, device=device
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
                )
            score_sum[y0:y1, x0:x1] += tile_score
            score_count[y0:y1, x0:x1] += 1.0

    return score_sum / np.maximum(score_count, 1.0)


def _compute_dino_heatmap(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    scale: float,
    device: str,
    max_patches: int = DEFAULT_MAX_PATCHES,
    tile_overlap: float = DEFAULT_TILE_OVERLAP,
) -> np.ndarray:
    """Runs one DINO view directly or in tiles when its patch grid is too large."""
    if max_patches <= 0:
        raise ValueError("max_patches must be greater than zero")

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
        )

    try:
        return _run_dino_heatmap(image, model, processor, scale=scale, device=device)
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
        )


def segment_anomalies(
    norm_score: np.ndarray,
    percentile_threshold: float = 99.5,
    adaptive_block_size: int = 21,
    morph_kernel_size: int = 3,
) -> np.ndarray:
    """Applies global percentile filtering, local adaptive edge matching, and morphology."""

    # Top-tail percentile mask
    threshold_val = np.percentile(norm_score, percentile_threshold)
    binary_mask = (norm_score >= threshold_val).astype(np.uint8) * 255

    # Local adaptive contrast refinement
    roi = (norm_score * 255).astype(np.uint8)
    core_mask = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block_size, -2
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

    Returns:
        filtered_mask: clean mask containing only the selected regions.
        regions_info: per-region metadata dicts:
            - 'label': connected-component ID.
            - 'area': area in pixels.
            - 'center': (x, y) coordinates guaranteed to be inside the region.
            - 'max_inscribed_radius': maximum Euclidean distance to the boundary.
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

            regions_info.append(
                {
                    "label": label,
                    "area": area,
                    "center": max_loc,  # (column_x, row_y) inside the area
                    "max_inscribed_radius": float(max_val),
                }
            )

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
    )
    final_mask = segment_anomalies(
        norm_score, percentile_threshold=percentile_threshold
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
) -> np.ndarray:
    """Runs DINO on one image view; returns binary anomaly mask at native image size."""
    scale = _view_scale(image, long_side, resolution_scale)
    norm_score = _compute_dino_heatmap(
        image,
        model,
        processor,
        scale=scale,
        device=device,
        max_patches=max_patches,
    )
    return segment_anomalies(norm_score, percentile_threshold=percentile_threshold)


def run_dino_view(
    image: Image.Image,
    model: AutoModel,
    processor: AutoImageProcessor,
    long_side: int = 1024,
    resolution_scale: float = 1.0,
    percentile_threshold: float = 99.5,
    device: Optional[str] = None,
    max_patches: int = DEFAULT_MAX_PATCHES,
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
    )
    final_mask = segment_anomalies(
        norm_score, percentile_threshold=percentile_threshold
    )
    return final_mask, norm_score, cropped_img


def _interest_zones(
    mask: np.ndarray,
    margin: float = 0.3,
    min_area: int = 30,
    max_zone_fraction: float = 0.8,
) -> list[tuple[int, int, int, int]]:
    """Binary mask -> expanded, merged bounding boxes of dense regions (native coords).
    Small components are dropped before expansion so noise specks never form zones."""
    h, w = mask.shape
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

    # greedy union of overlapping boxes until fixpoint
    zones = []
    for box in raw:
        merged = False
        for z in zones:
            if box[0] < z[2] and box[2] > z[0] and box[1] < z[3] and box[3] > z[1]:
                z[0], z[1] = min(z[0], box[0]), min(z[1], box[1])
                z[2], z[3] = max(z[2], box[2]), max(z[3], box[3])
                merged = True
                break
        if not merged:
            zones.append(box)

    return [
        (x0, y0, x1, y1)
        for x0, y0, x1, y1 in zones
        if (x1 - x0) * (y1 - y0) <= max_zone_fraction * w * h
    ]


def _merge_points(
    points: list[tuple[int, int, float, int]],
) -> list[tuple[int, int, float, int]]:
    """Greedy merge of points whose inscribed circles overlap; keeps the deepest level."""
    kept: list[list] = []
    for x, y, r, lvl in sorted(points, key=lambda p: -p[3]):
        for k in kept:
            if (x - k[0]) ** 2 + (y - k[1]) ** 2 < (r + k[2]) ** 2:
                # ponytail: O(n^2) greedy, fine for <100 pts/frame; DBSCAN if dense scenes appear
                k[0] = (k[0] + x) // 2
                k[1] = (k[1] + y) // 2
                k[2] = max(k[2], r)
                k[3] = max(k[3], lvl)
                break
        else:
            kept.append([x, y, r, lvl])
    return [tuple(k) for k in kept]


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
) -> List[Dict]:
    """
    Coarse-to-fine zoom detection with an optional relative resolution multiplier.
    DINO runs on a <=long_side view, multiplied by resolution_scale, then crops
    interest zones from the ORIGINAL pixels and re-detects until the view is
    native and <=long_side. Oversized scaled views are processed in overlapping
    tiles. Points are merged across levels and returned normalized.

    Returns list of dicts with keys:
        - 'x', 'y': normalized [0,1] center in original-image coords.
        - 'radius': max inscribed radius, normalized by max(W, H).
        - 'level': recursion depth the point was last confirmed at (deeper = finer).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image = remove_letterbox(image)
    W, H = image.size
    points: list[tuple[int, int, float, int]] = []

    def recurse(img: Image.Image, ox: int, oy: int, depth: int) -> None:
        w, h = img.size
        feed_native = max(w, h) <= long_side
        mask = _dino_pass(
            img, model, processor, long_side, device,
            percentile_threshold=percentile_threshold,
            resolution_scale=resolution_scale,
            max_patches=max_patches,
        )
        _, regions = extract_salient_regions(
            mask, min_absolute_pixels=min_object_pixels
        )
        for reg in regions:
            cx, cy = reg["center"]
            points.append((ox + cx, oy + cy, reg["max_inscribed_radius"], depth))
        if feed_native or depth >= max_levels:
            return
        for zx0, zy0, zx1, zy1 in _interest_zones(mask, margin=margin):
            recurse(img.crop((zx0, zy0, zx1, zy1)), ox + zx0, oy + zy0, depth + 1)

    recurse(image, 0, 0, 0)

    return [
        {"x": x / W, "y": y / H, "radius": r / max(W, H), "level": lvl}
        for x, y, r, lvl in _merge_points(points)
    ]


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
