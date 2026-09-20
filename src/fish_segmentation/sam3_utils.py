from typing import Any, List, Optional


def propagate_in_video(predictor, session_id, start_frame_index=None):
    """Streams `propagate_in_video` requests frame by frame and collects outputs.

    ``start_frame_index`` is required for tracker-only sessions (point prompts
    without a text/box prompt): the vendored SAM3 selection of the processing
    order otherwise raises "No prompts are received on any frames". Pass the
    frame where the points were added.
    """
    request = dict(
        type="propagate_in_video",
        session_id=session_id,
    )
    if start_frame_index is not None:
        request["start_frame_index"] = start_frame_index

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
