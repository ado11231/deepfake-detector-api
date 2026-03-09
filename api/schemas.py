from pydantic import BaseModel

# Defines the shape of the API response using Pydantic.
# Ensures every prediction response always contains a label and confidence score.
class PredictionResponse(BaseModel):
    label: str
    confidence: float
