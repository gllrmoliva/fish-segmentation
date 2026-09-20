"""CPU-only checks for the SAM3 request wrappers (no model, no GPU)."""

from fish_segmentation.sam3_utils import add_point_prompt, propagate_in_video


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
    }

    propagate_in_video(predictor, "sid")
    assert "start_frame_index" not in predictor.requests[1]
