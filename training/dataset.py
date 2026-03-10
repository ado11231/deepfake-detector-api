import os
from torch.utils.data import Dataset
from PIL import Image

# Dataset class for loading real and fake face images from two folders.
# Builds a path/label list on init, returns dataset size, and loads images as tensors by index.
class DeepfakeDataset(Dataset):

    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.data = []

        real_folder = os.path.join(self.path,  "real")
        for filename in os.listdir(real_folder):
            full_path = os.path.join(real_folder, filename)
            self.data.append((full_path, 0))

        fake_folder =os.path.join(self.path, "fake")
        for filename in os.listdir(fake_folder):
            full_path = os.path.join(fake_folder, filename)
            self.data.append((full_path, 1))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path, label = self.data[index]
        image = Image.open(path)
        tensor = self.transform(image)

        return tensor, label