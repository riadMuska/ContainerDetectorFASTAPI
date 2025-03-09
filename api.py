from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import io
import numpy as np
import cv2
from PIL import Image
import base64

app = FastAPI()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_model.pt', force_reload=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)

    img_cv = np.array(image)

    container_count = 0

    for det in results.xyxy[0]:
        if len(det) == 6:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
        else:
            x1, y1, x2, y2, conf = det
            cls = -1

        label = f"{model.names[cls] if cls != -1 else 'Unknown'} {conf:.2f}"

        cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_cv, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        container_count += 1

    _, img_encoded = cv2.imencode(".png", img_cv)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return JSONResponse(content={
        "image": img_base64,
        "count": container_count
    })
