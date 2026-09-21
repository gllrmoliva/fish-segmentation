from typing import Any, List, Optional

import numpy as np


def propagate_in_video(
    predictor,
    session_id,
    start_frame_index=None,
    propagation_direction="both",
    max_frame_num_to_track=None,
):
    """Streams `propagate_in_video` requests frame by frame and collects outputs.

    ``start_frame_index`` is required for tracker-only sessions (point prompts
    without a text/box prompt): the vendored SAM3 selection of the processing
    order otherwise raises "No prompts are received on any frames". Pass the
    frame where the points were added.

    ``propagation_direction`` is one of ``"both"``, ``"forward"`` or
    ``"backward"``. In the incremental DINO→SAM3 loop, after adding points at
    frame k, call ``"backward"`` (fills 0..k-1) and then ``"forward"`` (fills
    k..end) explicitly: a single ``"both"`` call does NOT backfill newly added
    objects — the reverse pass degenerates into a cached fetch because of the
    SAM3 action history (verified in the vendored
    ``parse_action_history_for_propagation``).
    """
    request = dict(
        type="propagate_in_video",
        session_id=session_id,
        propagation_direction=propagation_direction,
    )
    if start_frame_index is not None:
        request["start_frame_index"] = start_frame_index
    if max_frame_num_to_track is not None:
        request["max_frame_num_to_track"] = max_frame_num_to_track

    outputs_per_frame = {}
    for response in predictor.handle_stream_request(request=request):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def add_text_prompt(predictor, session_id, frame_index, text) -> dict[str, Any]:
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_index,
            text=text,
        )
    )
    return response["outputs"]


def add_point_prompt(
    predictor,
    session_id: str,
    frame_index: int,
    points: List[List[float]],
    labels: Optional[List[int]] = None,
    obj_id: Optional[int] = None,
    rel_coordinates: bool = True,
) -> dict[str, Any]:
    """Adds a tracker point prompt on one frame (one ``obj_id`` per instance).

    ``points`` is a list of ``[x, y]`` and ``labels`` defaults to all-positive
    (1). With ``rel_coordinates=True`` (SAM3 default) coordinates are relative
    [0, 1] of the original frame — the convention returned by
    ``run_zoom_detector``. Point prompts cannot be mixed with a text or box
    prompt in the same call; call again with the same ``obj_id`` to refine an
    existing object, or with a new ``obj_id`` to start another one.
    """
    if labels is None:
        labels = [1] * len(points)
    request = dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_index,
        points=points,
        point_labels=labels,
        rel_coordinates=rel_coordinates,
    )
    if obj_id is not None:
        request["obj_id"] = obj_id
    response = predictor.handle_request(request=request)
    return response["outputs"]


def point_in_existing_mask(
    det: dict,
    frame_outputs: Optional[dict],
    mask_margin_px: int = 0,
) -> bool:
    """True when a detection center falls inside an already tracked mask.

    ``frame_outputs`` is the SAM3 outputs dict of one frame (``out_binary_masks``
    at session resolution); ``mask_margin_px`` widens every existing mask by that
    many pixels (square window) so a point on a tracked animal's fin/edge still
    counts as covered. Points outside the frame are treated as covered (they
    cannot start a masklet).
    """
    if not frame_outputs:
        return False
    masks = frame_outputs.get("out_binary_masks")
    if masks is None:
        return False
    masks = np.asarray(masks)
    if masks.size == 0 or masks.ndim != 3:
        return False

    height, width = masks.shape[-2:]
    x = int(round(float(det["x"]) * width))
    y = int(round(float(det["y"]) * height))
    if not (0 <= x < width and 0 <= y < height):
        return True

    margin = max(0, int(mask_margin_px))
    y0, y1 = max(0, y - margin), min(height, y + margin + 1)
    x0, x1 = max(0, x - margin), min(width, x + margin + 1)
    return bool(masks[:, y0:y1, x0:x1].any())


def select_new_points(
    detections: List[dict],
    frame_outputs: Optional[dict] = None,
    min_score: Optional[float] = None,
    min_solidity: Optional[float] = None,
    min_area: Optional[float] = None,
    mask_margin_px: int = 0,
    dedupe_overlap: float = 0.3,
) -> List[dict]:
    """Filters DINO detections down to points that should start new masklets.

    Rejects (a) detections below the quality thresholds, (b) detections whose
    center is already covered by a tracked mask on that frame (see
    ``point_in_existing_mask``) and (c) near-duplicates among the survivors
    (bbox IoU above ``dedupe_overlap``), keeping the highest ``score_mean`` of
    each duplicate group. Returns the surviving detection dicts (the input dicts
    unchanged, so extra keys survive).
    """
    candidates = [
        det
        for det in detections
        if (min_score is None or det["score_mean"] >= min_score)
        and (min_solidity is None or det["solidity"] >= min_solidity)
        and (min_area is None or det["area"] >= min_area)
        and not point_in_existing_mask(det, frame_outputs, mask_margin_px)
    ]

    candidates.sort(key=lambda det: det["score_mean"], reverse=True)
    kept: List[dict] = []
    for det in candidates:
        if any(_bbox_iou(det["bbox"], other["bbox"]) > dedupe_overlap for other in kept):
            continue
        kept.append(det)
    return kept


def next_obj_id(outputs_per_frame: dict) -> int:
    """Smallest unused ``obj_id`` across a frame→outputs mapping (0 if empty)."""
    max_id = -1
    for outputs in outputs_per_frame.values():
        if not outputs:
            continue
        obj_ids = outputs.get("out_obj_ids")
        if obj_ids is None or len(obj_ids) == 0:
            continue
        max_id = max(max_id, int(np.max(obj_ids)))
    return max_id + 1


def _to_numpy(value):
    """Converts a (possibly CUDA/bf16) torch tensor to numpy, leaves arrays alone."""
    if value is None:
        return None
    for attr in ("detach", "cpu"):
        if hasattr(value, attr):
            value = getattr(value, attr)()
    if hasattr(value, "float"):  # bf16/fp16 tensors have no numpy dtype
        value = value.float()
    return np.asarray(value)


def segment_boxes(processor, image, boxes, label: bool = True) -> List[dict]:
    """Segments normalized boxes with the SAM3 IMAGE processor, one at a time.

    ``processor`` is a ``Sam3Processor`` (image-model path, see
    ``notebooks/05_smoke_test.ipynb``); this wrapper is duck-typed, so tests can
    pass a fake. ``boxes`` are ``[cx, cy, w, h]`` normalized [0, 1] (SAM3's
    geometric-prompt convention — convert DINO ``[x0, y0, x1, y1]`` boxes with
    ``_bbox_to_cxcywh``). Every box is reset before the next one because the
    processor accumulates geometric prompts in the same state.

    Returns one dict per box: ``{'mask': bool HxW or None, 'score': float}``,
    keeping the best (highest-score) mask of each box.
    """
    state = processor.set_image(image)
    results = []
    for box in boxes:
        out = processor.add_geometric_prompt(box=list(box), label=label, state=state)
        masks = _to_numpy(out.get("masks"))
        scores = _to_numpy(out.get("scores"))

        best = None
        if masks is not None and masks.size and scores is not None and scores.size:
            if masks.ndim == 4:  # (N, 1, H, W)
                masks = masks[:, 0]
            elif masks.ndim == 2:  # (H, W)
                masks = masks[None]
            scores = scores.reshape(-1)
            if masks.ndim == 3 and masks.shape[0] > 0 and scores.size > 0:
                count = min(masks.shape[0], scores.size)
                index = int(np.argmax(scores[:count]))
                best = {
                    "mask": masks[index].astype(bool),
                    "score": float(scores[index]),
                }
        results.append(best or {"mask": None, "score": 0.0})
        processor.reset_all_prompts(state)
    return results


def _bbox_to_cxcywh(box) -> List[float]:
    """Normalized ``[x0, y0, x1, y1]`` -> clamped ``[cx, cy, w, h]``."""
    x0 = min(max(float(box[0]), 0.0), 1.0)
    y0 = min(max(float(box[1]), 0.0), 1.0)
    x1 = min(max(float(box[2]), 0.0), 1.0)
    y1 = min(max(float(box[3]), 0.0), 1.0)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]


def detection_erase_mask(
    detections: List[dict],
    image,
    processor,
    max_area_fraction: float = 0.25,
    min_score: float = 0.5,
    fallback_shape: str = "bbox",
    fallback_radius_scale: float = 1.5,
) -> np.ndarray:
    """Erase mask for DINO detections: SAM3 box segmentation + region fallback.

    The signature matches ``run_dino_erase_loop``'s ``mask_fn(detections, image)``
    convention, so the loop takes it directly as
    ``partial(detection_erase_mask, processor=processor, ...)``.

    Runs ``segment_boxes`` with one box per detection (the normalized bbox) and
    falls back to ``dinov3.mask_from_detections`` when SAM3 returns no mask, a
    mask below ``min_score`` or one bigger than ``max_area_fraction`` of the
    frame — a class-agnostic box prompt on open water often segments the whole
    crop, which would erase real context. Returns a bool mask at the image
    resolution (the union of accepted masks and fallback regions).
    """
    from fish_segmentation.dinov3 import mask_from_detections

    width, height = image.size
    prompt_indices = []
    boxes = []
    for index, det in enumerate(detections):
        if det.get("bbox") is None:
            continue
        prompt_indices.append(index)
        boxes.append(_bbox_to_cxcywh(det["bbox"]))

    results = segment_boxes(processor, image, boxes) if boxes else []
    result_by_det = dict(zip(prompt_indices, results))

    frame_area = float(width * height)
    union = np.zeros((height, width), dtype=bool)
    fallback = []
    for index, det in enumerate(detections):
        result = result_by_det.get(index)
        accepted = False
        if result is not None and result["mask"] is not None:
            area = int(np.count_nonzero(result["mask"]))
            if result["score"] >= min_score and 0 < area <= max_area_fraction * frame_area:
                union |= result["mask"]
                accepted = True
        if not accepted:
            fallback.append(det)

    if fallback:
        union |= mask_from_detections(
            fallback,
            (width, height),
            shape=fallback_shape,
            radius_scale=fallback_radius_scale,
        )
    return union


def _bbox_iou(a, b) -> float:
    """IoU of two normalized ``[x0, y0, x1, y1]`` boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
