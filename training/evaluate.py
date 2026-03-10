# Loads the trained model and runs it against the validation set.
# Calculates and prints accuracy, precision, recall and confusion matrix.
# Splits dataset 80/20, runs training and validation loops for each epoch,
# prints loss after each epoch and saves final weights to model/detector.pth.
import torch
from torch.utils.data import DataLoader, random_split
from training.dataset import DeepfakeDataset
from core.model import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torchvision import transforms

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

dataset = DeepfakeDataset("data/", transform)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

def evaluate():
    model = load_model("model/detector.pth", device)
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            predicted = (predictions > 0.5).int()

            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{matrix}")

if __name__ == "__main__":
    evaluate()