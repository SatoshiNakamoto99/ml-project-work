from torch import nn
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
"""
Transfer learning from ResNet50 modified version to fire classification used in
Deep Convolutional Neural Networks for Fire Detection in Images (2017)
"""

def resnet_sharma(classes):
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = resnet.fc.in_features
    # modify the last layer to add another FC
    resnet.fc = nn.Linear(num_ftrs, 4096)
    # freeze for transfer learning
    freeze_layer = 9

    for i, child in enumerate(resnet.children()):
        if i < freeze_layer:
            for param in child.parameters():
                param.requires_grad = False

    return nn.Sequential(
        resnet,
        nn.Linear(4096, classes)
    )
# end resnet_sharma


def resnet_sharma_more_params(classes):
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = resnet.fc.in_features
    # modify the last layer to add another FC
    resnet.fc = nn.Linear(num_ftrs, 4096)
    # freeze for transfer learning
    freeze_layer = 7

    for i, child in enumerate(resnet.children()):
        if i < freeze_layer:
            for param in child.parameters():
                param.requires_grad = False

    return nn.Sequential(
        resnet,
        nn.Linear(4096, classes)
    )