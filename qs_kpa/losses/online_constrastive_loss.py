import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineContrastiveLoss(nn.Module):
    def __init__(self, distance_metric, margin: float = 0.5):
        """
        Online Contrastive loss. Similar to ConstrativeLoss.

        It selects hard positive (positives that are far apart) and hard negative pairs
        (negatives that are close) and computes the loss only for these pairs. Often yields
        better performances than  ConstrativeLoss.
        """
        super(OnlineContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distances = self.distance_metric(output1, output2)
        negs = distances[target == 0]
        poss = distances[target == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss.mean()
