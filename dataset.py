import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import random
from random import sample, random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from typing import List, Optional
import torchvision.transforms.functional as F
import os


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


class JigsawDataset(data.Dataset):
    def __init__(
            self,
            names: List[str],
            labels: List[int],
            path_dataset: str,
            jig_classes=30, img_transformer: Optional[transforms.Compose] = None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.image_transformer = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])

        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.image_resize = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor()
        ])

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self.augment_tile(tile)
        return tile

    def __getitem__(self, index):
     # Get and process an image
        image_path = self.data_path + "/" + self.names[index]
        img = Image.open(image_path).convert("RGB")
        if self.image_transformer:
          img = self.image_transformer(img)
        
        
       # if img.size[0] != 255:
        #  img = self.image_resize(img)

        # 9 is number of grids
        """
        tiles = [None] * 9
        for n in range(9):
            tiles[n] = self.get_tile(img, n)

        index = np.random.randint(len(self.permutations) + 1)
        if index == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[index - 1][t]]
                    for t in range(9)]

        data = torch.stack(data, 0)
        img = transforms.ToTensor()(img)
        """

        if np.random.rand() < 0.30:
            img = img.convert('LA').convert('RGB')

        if img.size[0] != 255:
            img = self.image_transformer(img)

        s = float(img.size[0]) / 3
        a = s / 2
        tiles = [None] * 9
        for n in range(9):
            i = n / 3
            j = n % 3
            c = [a * i * 2 + a, a * j * 2 + a]
            c = np.array([c[1] - a, c[0] - a, c[1] +
                          a + 1, c[0] + a + 1]).astype(int)
            tile = img.crop(c.tolist())
            tile = self.augment_tile(tile)
            # Normalize the patches indipendently to avoid low level features shortcut
            m, s = tile.view(
                3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
            s[s == 0] = 1
            #norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
            #tile = norm(tile)
            tiles[n] = tile

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data1 = torch.cat(data[0:3], 2)
        data2 = torch.cat(data[3:6], 2)
        data3 = torch.cat(data[6:9], 2)
        data = torch.cat((data1, data2, data3), 1)
        
        
        img = transforms.ToTensor()(img)
        
        

        #show([img, data])

        return img, int(self.labels[index]), data, int(order)

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


class JigsawTestDataset(data.Dataset):
    def __init__(
        self,
        names: List[str],
        labels: List[int],
        path_dataset: str,
        jig_classes=30, img_transformer: Optional[transforms.Compose] = None,
    ):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.image_transformer = img_transformer
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.grid_size = 3
        self.image_resize = transforms.Compose([
            transforms.Resize(256, Image.BILINEAR),
            transforms.CenterCrop(255)])
        self.augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor()
        ])

        def get_tile(self, img, n):
            w = float(img.size[0]) / self.grid_size
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self.augment_tile(tile)
            return tile

    def __getitem__(self, index):
        # Get and process an image
        image_path = self.data_path + "/" + self.names[index]
        img = Image.open(image_path).convert("RGB")
        if self.image_transformer:
            img = self.image_transformer(img)

        if img.size[0] != 255:
            img = self.image_resize(img)

        # 9 is number of grids
        tiles = [None] * 9
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        index = random.randrange(len(self.permutations) + 1)
        if index == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[index - 1][t]]
                    for t in range(n_grids)]
      
        data = torch.stack(data, 0)

        return img, int(self.labels[index]), data, int(index)

    def __len__(self):
        return len(self.names)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


# sphinx_gallery_thumbnail_path = "../../gallery/assets/visualization_utils_thumbnail.png"


# plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        img = transforms.ToTensor()(img)
        img = torch.squeeze(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # write image to file
        plt.savefig(f"figs/{np.random.randint(1,10000) + i}.png")
        plt.cla()
    plt.close(fix)
