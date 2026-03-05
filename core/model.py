import torch
import torch.nn as nn
from torchvision import models

#Create model using efficient net model, iterate through the weights and turn off gradients, get the input size and wrap it in a
#sequential container with 3 functions; to turn off 30% nuerons, allow one input and one output and to return between 0 and 1. Return model
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    for weight in model.parameters():
        weight.requires_grad = False

    input_size = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p = 0.3),
        nn.Linear(in_features = input_size, out_features = 1),
        nn.Sigmoid()
    )

    return model

#Load model that takes weight file path and device (cpu or cuda). Calls build_model and loads the weights and bias into the load variable.
#Stores the bias and weights into a serialized dictionary. Turn on inferece mode and return model
def load_model(file, device):
    model = build_model()
    load = torch.load(file, map_location = device)
    model.load_state_dict(load)

    model.eval()

    return model


