from core.model import load_model
from dotenv import load_dotenv
import os
import torch
# Accepts an image upload, converts it to OpenCV format, detects and preprocesses the face,
# runs it through the model and returns a label and confidence score as a JSON response.
load_dotenv()
path = os.getenv("MODEL_PATH")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(path, device)