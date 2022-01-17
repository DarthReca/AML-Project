import torch.utils.data as data
from PIL import Image
from random import random
import random
import torchvision
import torchvision.transforms.functional as TF
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import torchvision.transforms as transforms


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


class JigSawDataset(data.Dataset):
    def __init__(
            self,
            names: List[str],
            labels: List[int],
            path_dataset: str,
            jig_classes=30, img_transformer: Optional[transforms.Compose] = None):
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
            transforms.Lambda(rgb_jittering),
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
            data = [tiles[self.permutations[order - 1][t]]
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
            transforms.Lambda(rgb_jittering),
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
            data = [tiles[self.permutations[order - 1][t]]
                    for t in range(n_grids)]

        data = torch.stack(data, 0)

        show([img, data])
        return img, int(self.labels[index]), data, int(index)

    def __len__(self):
        return len(self.names)


# sphinx_gallery_thumbnail_path = "../../gallery/assets/visualization_utils_thumbnail.png"


#plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # write image to file
        plt.savefig(f"figs/{random.randint(10000) + i}.png")
        plt.cla()
    plt.close(fix)
