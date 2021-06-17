import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative).

    The loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative.
    """

    def __init__(self, distance_metric, triplet_margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor: torch.Tensor, rep_pos: torch.Tensor, rep_neg: torch.Tensor) -> torch.Tensor:
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()
