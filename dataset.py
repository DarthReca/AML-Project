import torch.utils.data as data
from PIL import Image
from random import random
import random
import torchvision.transforms.functional as TF

"""
_dataset_info() -> filename , label
It will be used in data_helper to retrieve sample's name and associated label.

"""

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

#It return train_dataset
class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):
        #TODO: get item with image and image rotation

        return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)


#It return test_dataset
class TestDataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        #TODO: get item with image and image rotation
        return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)

