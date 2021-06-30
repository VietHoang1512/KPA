from enum import Enum

import torch.nn.functional as F


class SiameseDistanceMetric(Enum):
    """
    The metric for the embedding silarity measurement
    """

    def EUCLIDEAN(x, y):
        return F.pairwise_distance(x, y, p=2)

    def MANHATTAN(x, y):
        return F.pairwise_distance(x, y, p=1)

    def COSINE_DISTANCE(x, y):
        return 1 - F.cosine_similarity(x, y)
