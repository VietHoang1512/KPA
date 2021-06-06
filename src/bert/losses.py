from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(vectors):
    distance_matrix = (
        -2 * vectors.mm(torch.t(vectors))
        + vectors.pow(2).sum(dim=1).view(1, -1)
        + vectors.pow(2).sum(dim=1).view(-1, 1)
    )
    return distance_matrix


class PairSelector:
    def __init__(self):
        """
        Implementation should return indices of positive pairs and negative pairs.

        Pass to compute Contrastive Loss return
        positive_pairs, negative_pairs.
        """

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    def __init__(self, balance=True):
        """
        Discards embeddings and generates all possible pairs given labels.

        If balance is True, negative pairs are a random sample to match the
        number of positive samples
        """
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[: len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    def __init__(self, cpu=True):
        """
        Creates all possible positive pairs.

        For negative pairs, pairs with smallest distance are taken into
        consideration, matching the number of positive pairs.
        """
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[: len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        """
        Contrastive loss Takes embeddings of two samples and a target
        label == 1 if samples are from the same class and label == 0 otherwise.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (
            target.float() * distances
            + (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)
        )
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin, pair_selector):
        """
        Online Contrastive loss Takes a batch of embeddings and corresponding labels.

        Pairs are generated using pair_selector object that take embeddings
        and targets and return indices of positive and negative pairs
        """
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(1).sqrt()
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()
