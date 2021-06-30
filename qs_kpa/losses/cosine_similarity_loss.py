import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self, loss_fct=nn.MSELoss()):
        """
        CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

        It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
        By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.
        """
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        similarity = torch.cosine_similarity(output1, output2)
        return self.loss_fct(similarity, target.view(-1))
