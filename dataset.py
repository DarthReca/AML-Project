import random
from typing import List, Optional, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def _dataset_info(txt_labels) -> Tuple[List[str], List[int]]:
    """It will be used in data_helper to retrieve sample's name and associated label."""
    with open(txt_labels, "r") as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(" ")
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

class FlipDataset(data.Dataset):
    def __init__(
        self,
        names: List[str],
        labels: List[int],
        path_dataset: str,
        img_transformer: Optional[transforms.Compose] = None,
    ):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.image_transformer = img_transformer

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        # Get and process image
        image_path = self.data_path + "/" + self.names[index]
        img = Image.open(image_path).convert("RGB")
        if self.image_transformer:
            img = self.image_transformer(img)

        
        # produce a random label 1 (True) or 0 (False)
        flip_flag = random.randrange(2)
        # image is either flipped or not that correspond to labels 0, 1.
        if flip_flag == 1:
            img_flip = torch.fliplr(img)
        else: 
            img_flip = img

        return img, self.labels[index], img_flip, flip_flag

    def __len__(self):
        return len(self.names)

class TestFlipDataset(data.Dataset):
    def __init__(
        self,
        names: List[str],
        labels: List[int],
        path_dataset: str,
        img_transformer: Optional[transforms.Compose] = None,
    ):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.image_transformer = img_transformer

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor, int]:
        # Get and process image
        image_path = self.data_path + "/" + self.names[index]
        img = Image.open(image_path).convert("RGB")
        if self.image_transformer:
            img = self.image_transformer(img)

        
        # produce a random label 1 (True) or 0 (False)
        flip_flag = random.randrange(2)
        # image is either flipped or not that correspond to labels 0, 1.
        if flip_flag == 1:
            img_flip = torch.fliplr(img)
        else: 
            img_flip = img

        return img, self.labels[index], img_flip, flip_flag

    def __len__(self):
        return len(self.names)