#pip install roboflow

import json

import os
from roboflow import Roboflow
import numpy as np
import supervision as sv

PROJECT_NAME = "streamlit-logo-detection"
VIDEO_FILE = "video.mp4"

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace().project(PROJECT_NAME)
model = project.version(1).model

job_id, signed_url, expire_time = model.predict_video(
    VIDEO_FILE,
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

with open("results.json", "w") as f:
    json.dump(results, f)

frame_offset = results["frame_offset"]
model_results = results[PROJECT_NAME]


def callback(scene: np.ndarray, index: int) -> np.ndarray:
    if index in frame_offset:
        detections = sv.Detections.from_inference(
            model_results[frame_offset.index(index)]
        )
    else:
        nearest = min(frame_offset, key=lambda x: abs(x - index))
        detections = sv.Detections.from_inference(
            model_results[frame_offset.index(nearest)]
        )

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        model_results[frame_offset.index(index)]["class_name"]
        for _
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=scene, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image


sv.process_video(
    source_path=VIDEO_FILE,
    target_path="output.mp4",
    callback=callback,
)