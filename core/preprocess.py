# Detects face in image, loads cascade classifier and converts to grayscale.
# Scans for faces requiring 5 confirmed neighbor detections before accepting a face region.
# If no face is found returns None, otherwise crops and returns the face using x, y, w, h coordinates.
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def detect_face(image):
    detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 5)

    if len(faces) == 0:
        return None
    else:
        x, y, w, h = faces[0] 
        sliced = image[y:y+h, x:x+w]

    return sliced

def preprocess_image(image):    
    valid_image = detect_face(image)

    if valid_image is None:
        return None
    
    valid_image = cv.cvtColor(valid_image, cv.COLOR_GRAY2RGB)

    pil_image = Image.fromarray(valid_image)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])

    tensor = transform(pil_image)
    tensor = tensor.unsqueeze(0)

    return tensor