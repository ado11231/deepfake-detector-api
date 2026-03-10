from fastapi import FastAPI, UploadFile, File, HTTPException
from api.schemas import PredictionResponse
from core.preprocess import preprocess_image
from core.predict import predict
from api.dependencies import model
import numpy as np
import cv2 as cv

# Accepts an image upload, converts it to OpenCV format, detects and preprocesses the face,
# runs it through the model and returns a label and confidence score as a JSON response.
app = FastAPI()

@app.post("/detect/")
async def get_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)

    cv_image = cv.imdecode(image_array, cv.IMREAD_COLOR)

    tensor = preprocess_image(cv_image)
    if tensor is None:
        raise HTTPException(status_code = 400, detail = "No Face Found")
    
    label, confidence = predict(model, tensor)
    
    return PredictionResponse(label = label, confidence = confidence)
