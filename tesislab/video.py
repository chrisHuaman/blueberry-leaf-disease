# pip install roboflow supervision numpy

import json
import os
from roboflow import Roboflow
import numpy as np
import supervision as sv

VIDEO_FILE = "./images/testvideo.mp4"
PROJECT_NAME = "drberry-leaf-detection-3hptt"
MODEL_VERSION = 1

# Inicializar Roboflow
rf = Roboflow(api_key="1lZZgNOL20KSm5TMIKox")
project = rf.workspace().project(PROJECT_NAME)
model = project.version(MODEL_VERSION).model

# Predecir el video
job_id, signed_url, expire_time = model.predict_video(
    VIDEO_FILE,
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

# Guardar resultados a un archivo JSON
with open("results.json", "w") as f:
    json.dump(results, f)

frame_offset = results["frame_offset"]
model_results = results[PROJECT_NAME]


# Función callback para procesar cada cuadro
def callback(scene: np.ndarray, index: int) -> np.ndarray:
    if index in frame_offset:
        inference_result = model_results[frame_offset.index(index)]
    else:
        nearest = min(frame_offset, key=lambda x: abs(x - index))
        inference_result = model_results[frame_offset.index(nearest)]

    detections = sv.Detections.from_inference(inference_result)

    # Cambiar BoundingBoxAnnotator a BoxAnnotator (si aplica)
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Generar etiquetas basadas en la clave correcta "class"
    labels = [
        prediction["class"]
        for prediction in inference_result["predictions"]
    ]

    # Anotar la imagen con las detecciones y etiquetas
    annotated_image = bounding_box_annotator.annotate(
        scene=scene, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image


# Procesar el video
sv.process_video(
    source_path=VIDEO_FILE,
    target_path="video_anotado.mp4",
    callback=callback,
)
