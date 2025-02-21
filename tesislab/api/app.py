from flask import Flask, request, jsonify
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="1lZZgNOL20KSm5TMIKox"
)

@app.route('/detect', methods=['POST'])
def detect_leaves():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró imagen'}), 400
    
    # Procesar imagen
    filestr = request.files['image'].read()
    npimg = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (600, 600))
    
    # Realizar inferencia
    try:
        result = CLIENT.infer(image, model_id="drberry-gen-2-hpfhd/1")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)