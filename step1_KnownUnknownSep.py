import argparse
from typing import Tuple

import torch
from optimizer_helper import get_optim_and_scheduler
from torch import nn
from torch.utils.data import DataLoader
from resnet import ResNet, Classifier
import os

# Implement Step1


def _do_epoch(
    args: argparse.Namespace,
    feature_extractor: ResNet,
    rot_cls: Classifier,
    obj_cls: Classifier,
    source_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Ideally we do an epoch and return all useful losses and accuracies.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace with various args.
    feature_extractor : ResNet
        Feature extractor for images.
    rot_cls : Classifier
        Rotation classifier (?).
    obj_cls : Classifier
        Object classifier (?).
    source_loader : DataLoader
        DataLoader of the source domain.
    optimizer : torch.optim.Optimizer
        Our gradient optimizer.
    device : torch.device
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
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    correct_classes = 0
    correct_rotations = 0
    for (data, class_label, rotated_data, rotated_label) in source_loader:
        data, class_label, rotated_data, rotated_label = (
            data.to(device),
            class_label.to(device),
            rotated_data.to(device),
            rotated_label.to(device),
        )
        optimizer.zero_grad()

        # Extract features
        original_features = feature_extractor(data)
        rotated_features = feature_extractor(rotated_data)
        # Pass features to classifiers
        class_scores = obj_cls(original_features)
        # Here we have to concatenate tensors as suggested by the architecture
        rotation_scores = rot_cls(torch.cat([original_features, rotated_features], 1))

        # Now we can check the losses
        class_loss = criterion(class_scores, class_label)
        rot_loss = criterion(rotation_scores, rotated_label)

        loss = class_loss + args.weight_RotTask_step1 * rot_loss

        loss.backward()

        optimizer.step()

        # Find which is the index that corresponds to the highest "probability"
        class_prediction = torch.argmax(class_scores, dim=1)
        rotation_prediction = torch.argmax(rotation_scores, dim=1)

        # Update counters
        correct_classes += torch.sum(class_prediction == class_label).item()
        correct_rotations += torch.sum(rotation_prediction == rotated_label).item()

    acc_cls = correct_classes / len(source_loader.dataset)
    acc_rot = correct_rotations / len(source_loader.dataset)

    return class_loss, acc_cls, rot_loss, acc_rot


def step1(
    args: argparse.Namespace,
    feature_extractor: ResNet,
    rot_cls: Classifier,
    obj_cls: Classifier,
    source_loader: DataLoader,
    device: torch.device,
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
        if epoch % 10 == 0:
            if not os.path.isdir("weights"):
                os.mkdir("weights")
            torch.save(
                feature_extractor.state_dict(), f"weights/feature_extractor_{epoch}.pth"
            )
            torch.save(obj_cls.state_dict(), f"weights/object_classifier_{epoch}.pth")
            torch.save(rot_cls.state_dict(), f"weights/rotation_classifier_{epoch}.pth")
        scheduler.step()
