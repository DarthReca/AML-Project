import torch
import torch.utils.data as data
from torchvision import transforms

from dataset import Dataset, TestDataset, _dataset_info

"""
get_train_dataloader(args,txt_file) 
Combines a dataset and a sampler, and provides an iterable over the given dataset. 
Parameters:
A dataset object (map style)
args:
path_dataset
batch_size

txt_file:
(source_path_file/target_path_file)


"""
# Iteration ovetr test_dataset
def get_train_dataloader(args, txt_file):
    """
    1.train_dataset (Dataset) : dataset from which to load the data.
    2.batch_size (int, optional) : how many samples per batch to load (default: 1).
    3.shuffle (bool, optional) : set to True to have the data reshuffled at every epoch (default: False).
    4.num_workers (int, optional) : how many subprocesses to use for data loading.
    0 means that the data will be loaded in the main process. (default: 0)
    5.pin_memory (bool, optional) : If True, the data loader will copy Tensors
    into CUDA pinned memory before returning them.
    If your data elements are a custom type, or your collate_fn
    returns a batch that is a custom type, see the example below.

    6.drop_last (bool, optional) : set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset
    is not divisible by the batch size, then the last batch will be smaller. (default: False)
    """

    img_transformer = get_train_transformers(args)
    name_train, labels_train = _dataset_info(txt_file)
    train_dataset = Dataset(
        name_train, labels_train, args.path_dataset, img_transformer=img_transformer
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return loader


# provide iteration over test_dataset
def get_val_dataloader(args, txt_file):

    names, labels = _dataset_info(txt_file)
    img_tr = get_test_transformer(args)
    test_dataset = TestDataset(names, labels, args.path_dataset, img_transformer=img_tr)
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return loader


# Used by get_train_dataloader()
def get_train_transformers(args):
    # Crop a random portion of image and resize it to a given size.
    """
    params:
    Size:image_size(h,w).
    Scale:Specifies the lower and upper bounds for the random area of the crop, before resizing.
    The scale is defined with respect to the area of the original image.

    """
    img_tr = [
        transforms.RandomResizedCrop(
            (int(args.image_size), int(args.image_size)),
            (args.min_scale, args.max_scale),
        )
    ]

    if args.jitter > 0.0:
        # ColorJitter->Randomly change the brightness, contrast, saturation and hue of an image.
        img_tr.append(
            transforms.ColorJitter(
                brightness=args.jitter,
                contrast=args.jitter,
                saturation=args.jitter,
                hue=min(0.5, args.jitter),
            )
        )
    if args.random_grayscale:
        img_tr.append(transforms.RandomGrayscale(args.random_grayscale))

    img_tr = img_tr + [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(img_tr)


# Used by get_val_dataloader()
def get_test_transformer(args):

    img_tr = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(img_tr)
