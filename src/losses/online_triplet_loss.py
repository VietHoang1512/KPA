import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: compare negative vs positive


class HardNegativeTripletLoss(nn.Module):
    def __init__(self, distance_metric, triplet_margin: float = 0.5):
        """
        For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet

        Margin should match the margin used in triplet loss.
        """
        super(HardNegativeTripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor: torch.Tensor, rep_pos: torch.Tensor, rep_neg: torch.Tensor) -> torch.Tensor:
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        neg_idx = distance_neg <= distance_neg.mean()

        distance_neg = distance_neg[neg_idx]
        distance_pos = distance_pos[neg_idx]

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()


class HardPositiveTripletLoss(nn.Module):
    def __init__(self, distance_metric, triplet_margin: float = 0.5):
        """
        For each negative pair, takes the hardest postive sample (with the lowest triplet loss value) to create a triplet.

        Margin should match the margin used in triplet loss.
        """
        super(HardPositiveTripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor: torch.Tensor, rep_pos: torch.Tensor, rep_neg: torch.Tensor) -> torch.Tensor:
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        pos_idx = distance_pos >= distance_pos.mean()

        distance_neg = distance_neg[pos_idx]
        distance_pos = distance_pos[pos_idx]

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()
