# CenterLoss by KaiyangZhou (see CENTER_LOSS_LICENSE)

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Parameters
    ----------
    num_classes: int
        number of classes.
    feat_dim: int
        feature dimension.

    References
    ----------
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes: int = 10, feat_dim: int = 2, use_gpu: bool = True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        print(self.use_gpu)

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim).cuda()
            )
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        """
        Parameters
        ----------
        x: Tensor
            feature matrix with shape (batch_size, feat_dim).
        labels: Tensor
            ground truth labels with shape (batch_size).
        """

        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss
