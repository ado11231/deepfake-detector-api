# Defines the shape of the API response using Pydantic.
# Ensures every prediction response always contains a label and confidence score.
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    confidence: float
