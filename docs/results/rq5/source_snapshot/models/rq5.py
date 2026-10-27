"""RQ5: matched nonlinear heads and auxiliary five-point supervision."""

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def image_encoder(pretrained):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    encoder = resnet18(weights=weights)
    encoder.fc = nn.Identity()
    return encoder


def nonlinear_head(num_attributes):
    return nn.Sequential(
        nn.Linear(544, 128), nn.ReLU(), nn.Linear(128, num_attributes)
    )


class RQ5RGBNonlinear(nn.Module):
    """N0: RGB features plus a fixed 32-dimensional zero placeholder."""

    def __init__(self, num_attributes=24, pretrained=True):
        super().__init__()
        self.image_encoder = image_encoder(pretrained)
        self.classifier = nonlinear_head(num_attributes)

    def forward(self, images):
        rgb = self.image_encoder(images)
        geometry = rgb.new_zeros((len(rgb), 32))
        return self.classifier(torch.cat((rgb, geometry), dim=1))


class RQ5PredictedGeometryNonlinear(nn.Module):
    """N1: predicted geometry; failed embeddings are masked AFTER the MLP."""

    def __init__(self, num_attributes=24, pretrained=True):
        super().__init__()
        self.image_encoder = image_encoder(pretrained)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU()
        )
        self.classifier = nonlinear_head(num_attributes)

    def encode_geometry(self, landmarks, geometry_valid):
        if landmarks.ndim != 2 or landmarks.shape[1] != 10:
            raise ValueError('landmarks must have shape [B, 10]')
        if geometry_valid.shape != (len(landmarks),) or geometry_valid.dtype != torch.bool:
            raise ValueError('geometry_valid must be boolean [B]')
        if not torch.isfinite(landmarks).all():
            raise ValueError('Coordinate placeholders must also be finite')
        geometry = self.geometry_encoder(landmarks)
        return geometry.masked_fill(~geometry_valid[:, None], 0.0)

    def forward(self, images, landmarks, geometry_valid):
        rgb = self.image_encoder(images)
        geometry = self.encode_geometry(landmarks, geometry_valid)
        return self.classifier(torch.cat((rgb, geometry), dim=1))


class RQ5LandmarkSupervised(nn.Module):
    """MTL: GT is a loss target, never an input to the attribute head."""

    def __init__(self, num_attributes=24, pretrained=True):
        super().__init__()
        self.image_encoder = image_encoder(pretrained)
        self.classifier = nn.Linear(512, num_attributes)
        self.landmark_head = nn.Linear(512, 10)

    def forward(self, images):
        rgb = self.image_encoder(images)
        return self.classifier(rgb), self.landmark_head(rgb)


def make_model(name, pretrained=True):
    classes = {'N0': RQ5RGBNonlinear, 'N1': RQ5PredictedGeometryNonlinear,
               'MTL': RQ5LandmarkSupervised}
    if name not in classes:
        raise ValueError(f'Unknown RQ5 model: {name}')
    return classes[name](pretrained=pretrained)
