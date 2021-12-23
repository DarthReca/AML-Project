import argparse
from typing import Literal, Tuple

import torch
from optimizer_helper import get_optim_and_scheduler
from torch import nn
from torch.utils.data import DataLoader

#### Implement Step1


def _do_epoch(
    args: argparse.Namespace,
    feature_extractor: nn.Module,
    rot_cls: nn.Module,
    obj_cls: nn.Module,
    source_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Literal["cuda", "cpu"],
) -> Tuple[float, float, float, float]:
    """
    Ideally we do an epoch and return all useful losses and accuracies.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace with various args.
    feature_extractor : nn.Module
        Feature extractor for images.
    rot_cls : nn.Module
        Rotation classifier (?).
    obj_cls : nn.Module
        Object classifier (?).
    source_loader : DataLoader
        DataLoader of the source domain (?).
    optimizer : torch.optim.Optimizer
        Our gradient optimizer.
    device : Literal["cuda", "cpu"]
        Where to put tensors.

    Returns
    -------
    class_loss: float
        Loss for object recognition
    class_acc: float
        Accuracy for object recognition
    rot_loss: float
        Loss for rotation recognition
    rot_acc: float
        Accuracy for rotation recognition
    """
    criterion = nn.CrossEntropyLoss()
    #TODO: rotate images
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    for it, (data, class_l, data_rot, rot_l) in enumerate(source_loader):
        data, class_l, data_rot, rot_l = (
            data.to(device),
            class_l.to(device),
            data_rot.to(device),
            rot_l.to(device),
        )
        optimizer.zero_grad()

        #TODO: compute the loss of the class and the rotation classification tasks
        class_loss = ...
        rot_loss = ...

        loss = class_loss + args.weight_RotTask_step1 * rot_loss

        loss.backward()

        optimizer.step()

        #TODO: store predicted class and rotation#TODO: compute the loss of the class and the rotation classification tasks
        _, cls_pred = ...
        _, rot_pred = ...

    #TODO: compute accuracy on class and rotation predictions
    acc_cls = ...
    acc_rot = ...

    return class_loss, acc_cls, rot_loss, acc_rot


def step1(
    args: argparse.Namespace,
    feature_extractor: nn.Module,
    rot_cls: nn.Module,
    obj_cls: nn.Module,
    source_loader: DataLoader,
    device: Literal["cuda", "cpu"],
) -> None:
    optimizer, scheduler = get_optim_and_scheduler(
        feature_extractor,
        rot_cls,
        obj_cls,
        args.epochs_step1,
        args.learning_rate,
        args.train_all,
    )

    for epoch in range(args.epochs_step1):
        print("Epoch: ", epoch)
        class_loss, acc_cls, rot_loss, acc_rot = _do_epoch(
            args, feature_extractor, rot_cls, obj_cls, source_loader, optimizer, device
        )
        print(
            "Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f"
            % (class_loss, acc_cls, rot_loss, acc_rot)
        )
        scheduler.step()
