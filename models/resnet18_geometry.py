"""ResNet18 with a five-landmark geometry branch."""

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18GeometryAttributes(nn.Module):
    """Fuse 512 image features with 32 geometry features."""

    def __init__(self, num_attributes=24, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.image_encoder = resnet18(weights=weights)

        # Keep the backbone's pooled image features.
        image_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        # Five normalized (x, y) points become 32 geometry features.
        self.geometry_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Predict attributes from the concatenated features.
        self.classifier = nn.Linear(image_features + 32, num_attributes)

    def forward(
        self,
        images: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if landmarks.ndim != 2 or landmarks.shape[1] != 10:
            raise ValueError("landmarks must have shape [B, 10]")
        if images.shape[0] != landmarks.shape[0]:
            raise ValueError("images and landmarks must share the batch size")

        image_features = self.image_encoder(images)
        geometry_features = self.geometry_encoder(landmarks)

        combined_features = torch.cat(
            [image_features, geometry_features],
            dim=1,
        )

        # Return logits for BCEWithLogitsLoss.
        return self.classifier(combined_features)
