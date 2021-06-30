import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, distance_metric, margin: float = 0.5):
        """
        Contrastive loss Takes embeddings of two samples and a target
        target == 1 if samples are from the same class and target == 0 otherwise.
        """
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distances = self.distance_metric(output1, output2)
        losses = 0.5 * (
            target.float() * distances.pow(2)
            + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps)).pow(2)
        )
        return losses.mean()
