import torch.nn as nn
import torchvision.models as models


def get_resnet18(pretrained=False, num_classes=10):

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
