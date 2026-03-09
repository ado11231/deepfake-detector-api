from core.model import load_model
from dotenv import load_dotenv
import os
import torch

load_dotenv()
path = os.getenv("MODEL_PATH")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(path, device)