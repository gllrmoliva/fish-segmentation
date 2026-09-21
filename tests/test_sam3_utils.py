"""CPU-only checks for the SAM3 request wrappers (no model, no GPU)."""

import numpy as np

from fish_segmentation.sam3_utils import (
    _bbox_to_cxcywh,
    _to_numpy,
    add_point_prompt,
    detection_erase_mask,
    next_obj_id,
    point_in_existing_mask,
    propagate_in_video,
    segment_boxes,
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


class FakeImageProcessor:
    """Minimal Sam3Processor stand-in for the image-model box wrapper."""

    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.prompts = []
        self.resets = 0

    def set_image(self, image):
        return {}

    def add_geometric_prompt(self, box, label, state):
        self.prompts.append((list(box), label))
        return self.outputs.pop(0)

    def reset_all_prompts(self, state):
        self.resets += 1


def test_segment_boxes_keeps_best_mask_and_resets():
    masks = np.zeros((2, 1, 6, 6), dtype=bool)
    masks[1, 0, 2:4, 2:4] = True
    processor = FakeImageProcessor(
        [{"masks": masks, "scores": np.array([0.1, 0.9])}]
    )

    results = segment_boxes(processor, image=None, boxes=[[0.5, 0.5, 0.4, 0.4]])

    assert results[0]["score"] == 0.9
    assert results[0]["mask"].shape == (6, 6)
    assert results[0]["mask"][2:4, 2:4].all()
    assert processor.prompts == [([0.5, 0.5, 0.4, 0.4], True)]
    assert processor.resets == 1


def test_segment_boxes_empty_output_is_none():
    processor = FakeImageProcessor(
        [{"masks": np.zeros((0, 1, 6, 6), bool), "scores": np.zeros(0)}]
    )

    results = segment_boxes(processor, image=None, boxes=[[0.5, 0.5, 0.1, 0.1]])

    assert results[0] == {"mask": None, "score": 0.0}
    assert processor.resets == 1


def test_bbox_to_cxcywh_clamps_and_orders():
    assert _bbox_to_cxcywh((-0.1, 0.2, 0.5, 0.9)) == [0.25, 0.55, 0.5, 0.7]
    assert _bbox_to_cxcywh((0.5, 0.9, 0.1, 0.2)) == [0.3, 0.55, 0.4, 0.7]


def test_to_numpy_casts_bf16_tensor():
    """Under bf16 autocast the SAM3 outputs are bf16 and .numpy() would reject them."""

    class FakeBf16Tensor:
        dtype = "torch.bfloat16"

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return np.array([0.25, 0.5], dtype=np.float32)

    assert np.allclose(_to_numpy(FakeBf16Tensor()), [0.25, 0.5])
    assert _to_numpy(None) is None
    plain = np.array([1.0, 2.0])
    assert _to_numpy(plain) is plain


def _image_100():
    from PIL import Image

    return Image.new("RGB", (100, 100))


def test_detection_erase_mask_falls_back_on_large_or_weak_masks():
    giant = np.zeros((1, 1, 100, 100), dtype=bool)
    giant[0, 0, :70, :70] = True  # 4900 px > 25% of 10000
    small = np.zeros((1, 1, 100, 100), dtype=bool)
    small[0, 0, 60:70, 60:70] = True

    processor = FakeImageProcessor(
        [
            {"masks": giant, "scores": np.array([0.9])},
            {"masks": small, "scores": np.array([0.9])},
        ]
    )
    detections = [
        _det(0.15, 0.15, bbox=(0.1, 0.1, 0.2, 0.2)),
        _det(0.65, 0.65, bbox=(0.6, 0.6, 0.7, 0.7)),
    ]

    union = detection_erase_mask(
        detections, _image_100(), processor, max_area_fraction=0.25, min_score=0.5
    )

    assert union[10:20, 10:20].all(), "rejected giant mask must fall back to the bbox"
    assert union[60:70, 60:70].all(), "accepted SAM3 mask must be kept"
    assert not union[:5, :5].any()

    # a low-confidence mask is rejected too
    processor = FakeImageProcessor([{"masks": small, "scores": np.array([0.1])}])
    union = detection_erase_mask(
        [_det(0.15, 0.15, bbox=(0.1, 0.1, 0.2, 0.2))],
        _image_100(),
        processor,
        min_score=0.5,
    )
    assert union[10:20, 10:20].all()


def test_detection_erase_mask_without_bbox_skips_sam3():
    processor = FakeImageProcessor([])

    union = detection_erase_mask(
        [{"x": 0.5, "y": 0.5, "radius": 0.1, "bbox": None}], _image_100(), processor
    )

    assert union.any()
    assert processor.prompts == []
