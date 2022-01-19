from typing import List, Union

import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls


class ResNet(nn.Module):
    """ResNet implements this architecture (in this case only ResNet18)."""

    def __init__(self, block: Union[BasicBlock, Bottleneck], layers: List[int]) -> None:
        # Various parameters
        self.inplanes = 64
        super(ResNet, self).__init__()
        # Layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Make 2 basic blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        # Make 2 basic blocks
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # Make 2 basic blocks
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # Make 2 basic blocks
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # Init layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Union[BasicBlock, Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def is_patch_based(self):
        return False

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Classifier(nn.Module):
    def __init__(self, input_size: int, classes: int):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Linear(input_size, classes)

    def forward(self, x: torch.Tensor):
        return self.class_classifier(x)


def resnet18_feat_extractor() -> ResNet:
    """
    Construct a ResNet-18 model.

    Notes
    -----
    See here to have a better idea of the architecture: https://learnopencv.com/wp-content/uploads/2020/03/resnet-18.png
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(model_zoo.load_url(
        model_urls["resnet18"]), strict=False)

    return model
