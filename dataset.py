import torch.utils.data as data
from PIL import Image
from random import random
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import torch


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

#It returns train_dataset
class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):
        #TODO: get item with image and image rotation
        #get the image path
          image_path = self.data_path + '/' + self.names[index]
          #produce a random label between 0,1,2,3
          img = Image.open(image_path).convert('RGB')

          index_rot = random.randrange(4)
          #The angles of rotation are 0°, 90°, 180°, 270° that correspond to labels 0, 1, 2, 3.

          if index_rot == 0:
              img_rot = img

          if index_rot == 1:
              img_rot = torch.rot90(img, k=1, dims=[1, 2]) #(img, k=1)

          if index_rot == 2:
              img_rot = torch.rot90(img, k=2, dims=[1, 2])

          if index_rot == 3:
              img_rot = torch.rot90(img, k=3, dims=[1, 2])
              #convert to tensor
          img_rot = TF.to_tensor(img_rot)
          return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)


#It returns test_dataset
class TestDataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        #TODO: get item with image and image rotation
          image_path = self.data_path + '/' + self.names[index]
          img = Image.open(image_path).convert('RGB')
          #produce a random label between 0,1,2,3
          index_rot = np.random.randint(4)
          #The angles of rotation are 0°, 90°, 180°, 270° that correspond to labels 0, 1, 2, 3.

          if index_rot == 0:
              img_rot = img

          if index_rot == 1:
              img_rot = np.rot90(img, k=1)

          if index_rot == 2:
              img_rot = np.rot90(img, k=2)

          if index_rot == 3:
              img_rot = np.rot90(img, k=3)

          img_rot = TF.to_tensor(img_rot)
        
          return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)

