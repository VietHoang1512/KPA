import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self, loss_fct=nn.MSELoss()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct

    def forward(self, output1, output2, target):
        similarity = torch.cosine_similarity(output1, output2)
        return self.loss_fct(similarity, target.view(-1))


class ContrastiveLoss(nn.Module):
    def __init__(self, distance_metric, margin: float):
        """
        Contrastive loss Takes embeddings of two samples and a target
        target == 1 if samples are from the same class and target == 0 otherwise.
        """
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = self.distance_metric(output1, output2)
        losses = 0.5 * (
            target.float() * distances.pow(2)
            + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps)).pow(2)
        )
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    def __init__(self, distance_metric, margin: float):
        """
        Online Contrastive loss. Similar to ConstrativeLoss.

        It selects hard positive (positives that are far apart) and hard negative pairs
        (negatives that are close) and computes the loss only for these pairs. Often yields
        better performances than  ConstrativeLoss.
        """
        super(OnlineContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin

    def forward(self, output1, output2, target):
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
