import os
HOME = os.getcwd()
print("I am here!", HOME)


# import subprocess
# subprocess.run(["¨pip", "install", "-r", "requirements.txt"])

import cv2
import supervision as sv
import matplotlib.pyplot as plt


image_file = "./images/original.jpg"
image_load = cv2.imread(image_file)
if image_load is None:
    raise FileNotFoundError(f"No se encontró el archivo: {image_file}")
image = cv2.resize(image_load, (600, 600))


# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="IIysMUQjOQynWBDBpm5d"
)

#result = CLIENT.infer(image, model_id="drberry-gen-2-hpfhd/1")
try:
    result = CLIENT.infer(image, model_id="drberry-gen-3-nzgyh/1")
except Exception as e:
    print(f"Error durante la inferencia: {e}")

# Procesar los resultados con Supervision
detections = sv.Detections.from_inference(result)

# Inicializar anotadores
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

# Extraer etiquetas de las predicciones
labels = [item["class"] for item in result["predictions"]]

# Dibujar cuadros y etiquetas sobre la imagen
annotated_image = image.copy()
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
