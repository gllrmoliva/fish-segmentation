"""CPU-only checks for the SAM3 request wrappers (no model, no GPU)."""

import numpy as np

from fish_segmentation.sam3_utils import (
    add_point_prompt,
    next_obj_id,
    point_in_existing_mask,
    propagate_in_video,
    select_new_points,
)


class FakePredictor:
    def __init__(self):
        self.requests = []

    def handle_request(self, request):
        self.requests.append(request)
        return {"outputs": {"ok": True}}


class FakeStreamPredictor:
    def __init__(self):
        self.requests = []

    def handle_stream_request(self, request):
        self.requests.append(request)
        yield {"frame_index": 0, "outputs": {"a": 1}}
        yield {"frame_index": 1, "outputs": {"a": 2}}


def test_add_point_prompt_builds_the_request():
    predictor = FakePredictor()

    out = add_point_prompt(predictor, "sid", 0, [[0.5, 0.25]], obj_id=3)

    assert out == {"ok": True}
    assert predictor.requests[0] == {
        "type": "add_prompt",
        "session_id": "sid",
        "frame_index": 0,
        "points": [[0.5, 0.25]],
        "point_labels": [1],
        "rel_coordinates": True,
        "obj_id": 3,
    }


def test_add_point_prompt_labels_and_optional_obj_id():
    predictor = FakePredictor()

    add_point_prompt(predictor, "sid", 1, [[0.1, 0.2]], labels=[0], obj_id=4)
    assert predictor.requests[0]["point_labels"] == [0]
    assert predictor.requests[0]["obj_id"] == 4

    add_point_prompt(predictor, "sid", 2, [[0.3, 0.4]])
    assert "obj_id" not in predictor.requests[1]
    assert predictor.requests[1]["point_labels"] == [1]


def test_propagate_in_video_passes_start_frame_index():
    predictor = FakeStreamPredictor()

    outputs = propagate_in_video(predictor, "sid", start_frame_index=3)

    assert outputs == {0: {"a": 1}, 1: {"a": 2}}
    assert predictor.requests[0] == {
        "type": "propagate_in_video",
        "session_id": "sid",
        "start_frame_index": 3,
        "propagation_direction": "both",
    }

    propagate_in_video(predictor, "sid")
    assert "start_frame_index" not in predictor.requests[1]


def test_propagate_in_video_direction_and_max_frames():
    predictor = FakeStreamPredictor()

    propagate_in_video(
        predictor,
        "sid",
        start_frame_index=10,
        propagation_direction="backward",
        max_frame_num_to_track=5,
    )

    assert predictor.requests[0] == {
        "type": "propagate_in_video",
        "session_id": "sid",
        "start_frame_index": 10,
        "propagation_direction": "backward",
        "max_frame_num_to_track": 5,
    }


def _det(x, y, score=0.9, solidity=0.5, area=100.0, bbox=None, radius=0.02):
    if bbox is None:
        bbox = (x - 0.02, y - 0.02, x + 0.02, y + 0.02)
    return {
        "x": x,
        "y": y,
        "score_mean": score,
        "solidity": solidity,
        "area": area,
        "bbox": bbox,
        "radius": radius,
    }


def _frame_outputs(masks):
    masks = np.asarray(masks, dtype=bool)
    return {
        "out_obj_ids": np.arange(masks.shape[0]),
        "out_binary_masks": masks,
    }


def test_point_in_existing_mask_containment_and_margin():
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, 2:5, 2:5] = True  # covers normalized x/y ~0.3

    inside = _det(0.3, 0.3)
    assert point_in_existing_mask(inside, _frame_outputs(masks))

    outside = _det(0.8, 0.8)
    assert not point_in_existing_mask(outside, _frame_outputs(masks))

    # just below the blob: outside with margin 0, inside with margin 2
    below = _det(0.3, 0.6)
    assert not point_in_existing_mask(below, _frame_outputs(masks), mask_margin_px=0)
    assert point_in_existing_mask(below, _frame_outputs(masks), mask_margin_px=2)


def test_point_in_existing_mask_without_outputs():
    det = _det(0.5, 0.5)
    assert not point_in_existing_mask(det, None)
    assert not point_in_existing_mask(det, {})
    assert not point_in_existing_mask(
        det, _frame_outputs(np.zeros((0, 10, 10), dtype=bool))
    )
    # off-frame points cannot start a masklet
    assert point_in_existing_mask(_det(1.2, 0.5), _frame_outputs(np.zeros((1, 10, 10))))


def test_select_new_points_dedupes_against_tracked_masks():
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, 2:5, 2:5] = True

    detections = [_det(0.3, 0.3, score=0.99), _det(0.8, 0.8, score=0.8)]

    kept = select_new_points(detections, frame_outputs=_frame_outputs(masks))

    assert [det["x"] for det in kept] == [0.8]


def test_select_new_points_quality_thresholds():
    detections = [
        _det(0.1, 0.1, score=0.2, solidity=0.9, area=500.0),
        _det(0.4, 0.4, score=0.9, solidity=0.05, area=500.0),
        _det(0.7, 0.7, score=0.9, solidity=0.9, area=5.0),
        _det(0.9, 0.9, score=0.9, solidity=0.9, area=500.0),
    ]

    kept = select_new_points(
        detections, min_score=0.5, min_solidity=0.3, min_area=50.0
    )

    assert len(kept) == 1
    assert kept[0]["x"] == 0.9


def test_select_new_points_nms_keeps_best_score():
    overlapping = [
        _det(0.5, 0.5, score=0.7, bbox=(0.4, 0.4, 0.6, 0.6)),
        _det(0.52, 0.52, score=0.95, bbox=(0.42, 0.42, 0.62, 0.62)),
    ]

    kept = select_new_points(overlapping, dedupe_overlap=0.3)

    assert len(kept) == 1
    assert kept[0]["score_mean"] == 0.95

    # a low overlap threshold keeps both
    assert len(select_new_points(overlapping, dedupe_overlap=0.9)) == 2


def test_next_obj_id_scans_every_frame():
    assert next_obj_id({}) == 0
    assert next_obj_id({0: None}) == 0
    assert next_obj_id({0: {"out_obj_ids": np.array([])}}) == 0
    assert (
        next_obj_id(
            {
                0: {"out_obj_ids": np.array([0, 1])},
                5: {"out_obj_ids": np.array([2])},
            }
        )
        == 3
    )
