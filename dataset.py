import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from typing import List, Optional
from itertools import product


def _dataset_info(txt_labels):
    with open(txt_labels, "r") as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(" ")
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


class JigsawDataset(data.Dataset):
    """This version works only with size 222, 222"""

    def __init__(
        self,
        names: List[str],
        labels: List[int],
        path_dataset: str,
        jig_classes=30,
        img_transformer: Optional[transforms.Compose] = None,
    ):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.image_transformer = img_transformer

        self.permutations = self.__retrieve_permutations(jig_classes)

    def __getitem__(self, index):
        # Get and process an image
        image_path = self.data_path + "/" + self.names[index]
        img = TF.resize(Image.open(image_path).convert("RGB"), [222, 222])

        tiles = [
            TF.to_tensor(TF.crop(img, 74 * row, 74 * column, 74, 74))
            for row, column in product(range(3), range(3))
        ]

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data1 = torch.cat(data[0:3], 2)
        data2 = torch.cat(data[3:6], 2)
        data3 = torch.cat(data[6:9], 2)
        data = torch.cat((data1, data2, data3), 1)

        img = transforms.ToTensor()(img)

        return img, int(self.labels[index]), data, int(order)

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load("permutations_%d.npy" % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


class JigsawTestDataset(data.Dataset):
    """This version works only with size 222, 222"""

    def __init__(
        self,
        names: List[str],
        labels: List[int],
        path_dataset: str,
        jig_classes=30,
        img_transformer: Optional[transforms.Compose] = None,
    ):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.image_transformer = img_transformer

        self.permutations = self.__retrieve_permutations(jig_classes)

    def __getitem__(self, index):
        # Get and process an image
        image_path = self.data_path + "/" + self.names[index]
        img = TF.resize(Image.open(image_path).convert("RGB"), [222, 222])

        tiles = [
            TF.to_tensor(TF.crop(img, 74 * row, 74 * column, 74, 74))
            for row, column in product(range(3), range(3))
        ]

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data1 = torch.cat(data[0:3], 2)
        data2 = torch.cat(data[3:6], 2)
        data3 = torch.cat(data[6:9], 2)
        data = torch.cat((data1, data2, data3), 1)

        img = transforms.ToTensor()(img)

        return img, int(self.labels[index]), data, int(index)

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load("permutations_%d.npy" % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        img = transforms.ToTensor()(img)
        img = torch.squeeze(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # write image to file
        plt.savefig(f"figs/{np.random.randint(1,10000) + i}.png")
        plt.cla()
    plt.close(fix)
