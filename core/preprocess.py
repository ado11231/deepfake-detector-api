import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

#Detects face in image, loads cascade classifier and runs image through it. Stores the faces in a list with a capacity of 5.
#If faces are not found, it returns none, if faces are found, its slices up into 4 regions, the face and distance from top and left edge
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

# Takes a raw image, detects and crops the face, converts to RGB, then applies
# resize, tensor conversion and normalization before returning a model-ready tensor.
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





        


