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


class Dataset(data.Dataset):
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
        # produce a random label between 0,1,2,3
        index_rot = random.randrange(4)
        # The angles of rotation are 0°, 90°, 180°, 270° that correspond to labels 0, 1, 2, 3.
        if index_rot == 0:
            img_rot = img
        elif index_rot == 1:
            img_rot = torch.rot90(img, k=1, dims=[1, 2])
        elif index_rot == 2:
            img_rot = torch.rot90(img, k=2, dims=[1, 2])
        elif index_rot == 3:
            img_rot = torch.rot90(img, k=3, dims=[1, 2])

        return img, self.labels[index], img_rot, index_rot

    def __len__(self):
        return len(self.names)


class TestDataset(data.Dataset):
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
        image_path = self.data_path + "/" + self.names[index]
        img = Image.open(image_path).convert("RGB")
        if self.image_transformer:
            img = self.image_transformer(img)
        # produce a random label between 0,1,2,3
        index_rot = random.randrange(4)
        # The angles of rotation are 0°, 90°, 180°, 270° that correspond to labels 0, 1, 2, 3.
        if index_rot == 0:
            img_rot = img
        elif index_rot == 1:
            img_rot = torch.rot90(img, k=1, dims=[1, 2])
        elif index_rot == 2:
            img_rot = torch.rot90(img, k=2, dims=[1, 2])
        elif index_rot == 3:
            img_rot = torch.rot90(img, k=3, dims=[1, 2])

        return img, self.labels[index], img_rot, index_rot

    def __len__(self):
        return len(self.names)
