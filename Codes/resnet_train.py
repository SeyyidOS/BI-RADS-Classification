import os
import pandas as pd
from torchvision.io import read_image

class BIRadsDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        veri = pd.read_excel(annotations_file)
        self.img_labels = veri['BIRADS KATEGORİSİ']
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label