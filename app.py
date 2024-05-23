from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
from ultralytics import YOLO
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = YOLO('yolov8n-seg.pt')

PowderBlue = np.array([176, 224, 230])
SlateBlue = np.array([106, 90, 205])
PeachPuff = np.array([255, 218, 185])
colors = [PowderBlue, SlateBlue, PeachPuff]

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = base64.b64decode(data['image'])
    color_flag = int(data['color_flag'])

    image = np.array(Image.open(io.BytesIO(image_data)))

    results = model(image, stream=True, classes=0)
    
    if color_flag is not None:
        for r in results:
            if r.masks is None:
                continue
            boxes = r.boxes.data
            masks = r.masks.data

            clss = boxes[:, 5]
            people_indices = torch.where(clss == 0)
            people_masks = masks[people_indices]
            people_mask = torch.any(people_masks, dim=0).int()

            background_mask = 1 - people_mask

            image[background_mask.bool()] = colors[color_flag]

    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': encoded_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
