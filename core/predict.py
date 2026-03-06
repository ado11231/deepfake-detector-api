import torch

#Runs a tensor through the model inside a no_grad context, extracts the raw score.
#Compares it against the 0.5 threshold, and returns a human readable label and confidence percentage.
def predict(model, tensor) -> tuple[str, float]:
    with torch.no_grad():
        output = model(tensor)

    score = output[0][0].item()

    threshold = 0.5
    
    if score >= threshold:
        label, confidence = "fake", round(score * 100, 2)
        return label, confidence
    else:
        label, confidence = "real", round((1 - score) * 100, 2)
        return label, confidence
        
        







