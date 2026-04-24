from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile

app = Flask(__name__)
model = YOLO('runs/detect/sign_detection/weights/best.pt')
class_names = {0: 'Stop', 1: 'Speed Limit', 2: 'Yield', 3: 'No Entry'}

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
        file.save(temp.name)
        image = cv2.imread(temp.name)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400
    results = model.predict(image)
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names.get(class_id, 'Unknown')
            xyxy = [float(x) for x in box.xyxy[0]]
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': xyxy
            })
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
