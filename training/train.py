# Training script for the deepfake detector model.
# Loads the dataset, splits into train/val sets, and runs the training loop for a set number of epochs.
# Each epoch trains on batches, calculates loss, backpropagates, and updates weights.
# Validates after each epoch to monitor performance, then saves the final weights to model/detector.pth.
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import random_split, DataLoader
from training.dataset import DeepfakeDataset
from core.model import build_model
from torchvision import transforms

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATA_PATH = "data/"

dataset = DeepfakeDataset(DATA_PATH, transform)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = build_model().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

def train(model, train_loader, val_loader, criterion, optimizer, device):
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(images)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.float().unsqueeze(1)
                
                predictions = model(images)
                loss = criterion(predictions, labels)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    train(model, train_loader, val_loader, criterion, optimizer, device)
    torch.save(model.state_dict(), "model/detector.pth")
    print("Model saved to model/detector.pth")