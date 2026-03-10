# Deepfake Detector

A REST API that detects whether a face image is real or AI-generated. Built with FastAPI and a fine-tuned EfficientNet-B0 model. Accepts an image upload and returns a prediction with a confidence score.

## Requirements

- Python 3.9+
- pip

## Setup

1. Clone the repo
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector

2. Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate

3. Install dependencies
   pip install -r requirements.txt

4. Add environment variables
   Create a .env file in the root with:
   MODEL_PATH=model/detector.pth
   CONFIDENCE_THRESHOLD=0.5

## Training

1. Download the 140k Real and Fake Faces dataset from Kaggle
2. Place it in a data/ folder structured as:
   data/
   ├── real/
   └── fake/

3. Run training
   python training/train.py

## Running the API

   uvicorn api.main:app --reload

API will be available at http://localhost:8000
Interactive docs at http://localhost:8000/docs

## Endpoints

POST /detect
- Accepts an image file upload
- Returns label (real/fake) and confidence score

Example response:
{
    "label": "fake",
    "confidence": 87.32
}

## Tech Stack

- FastAPI
- PyTorch
- EfficientNet-B0
- OpenCV
- Scikit-learn
