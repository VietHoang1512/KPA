from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners

from src.backbone.base_model import BaseModel
from src.pseudo_label.model_argument import PseudoLabelModelArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class PseudoLabelModel(BaseModel):
    def __init__(self, args: PseudoLabelModelArguments):
        """
        Pseudo Label Model.

        Args:
            args (PseudoLabelModelArguments): Pseudo Label Model Argument
        """
        super().__init__(args)

        if self.args.distance == "cosine":
            self.distance = distances.CosineSimilarity()
        elif self.args.distance == "euclidean":
            self.distance = distances.LpDistance()
        else:
            raise NotImplementedError(
                f"Embedding similarity function {self.args.distance} is not implemented yet. Must be `consine` or `euclidean`"
            )

        main_loss = losses.TupletMarginLoss(margin=self.args.margin, distance=self.distance)
        var_loss = losses.IntraPairVarianceLoss(distance=self.distance)
        self.loss_func = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
        self.mining_func = miners.MultiSimilarityMiner(epsilon=0.2, distance=self.distance)

    def criterion(self, embeddings, labels):
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss_func(embeddings, labels, indices_tuple)
        # loss = self.loss_func(embeddings, labels)
        return loss

    def forward(
        self,
        topic_input_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        topic_token_type_ids: torch.Tensor,
        stance: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        statements_encoded: Optional[torch.Tensor] = None,
        key_point_input_ids: Optional[torch.Tensor] = None,
        key_point_attention_mask: Optional[torch.Tensor] = None,
        key_point_token_type_ids: Optional[torch.Tensor] = None,
        argument_input_ids: Optional[torch.Tensor] = None,
        argument_attention_mask: Optional[torch.Tensor] = None,
        argument_token_type_ids: Optional[torch.Tensor] = None,
    ):
        if self.training:
            n_statements = statements_encoded[0].shape[0]

            stance_rep = self.fc_stance(stance).repeat(n_statements, 1)
            topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids).repeat(
                n_statements, 1
            )

            statement_bert_output = self._forward_text(
                statements_encoded[0][:, 0], statements_encoded[0][:, 1], statements_encoded[0][:, 2]
            )

            statement_rep = torch.cat([stance_rep, topic_bert_output, statement_bert_output], axis=1)
            statement_rep = self.fc_text(statement_rep)
            if self.args.normalize:
                statement_rep = F.normalize(statement_rep, p=2, dim=1)

            loss = self.criterion(statement_rep, label[0])

            return loss
        else:
            stance_rep = self.fc_stance(stance)
            topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)
            key_point_bert_output = self._forward_text(
                key_point_input_ids, key_point_attention_mask, key_point_token_type_ids
            )
            argument_bert_output = self._forward_text(
                argument_input_ids, argument_attention_mask, argument_token_type_ids
            )

            argument_rep = torch.cat([stance_rep, topic_bert_output, argument_bert_output], axis=1)
            argument_rep = self.fc_text(argument_rep)
            keypoint_rep = torch.cat([stance_rep, topic_bert_output, key_point_bert_output], axis=1)
            keypoint_rep = self.fc_text(keypoint_rep)
            if self.args.normalize:
                argument_rep = F.normalize(argument_rep, p=2, dim=1)
                keypoint_rep = F.normalize(keypoint_rep, p=2, dim=1)
            similarity = (self.args.margin - self.distance_metric(argument_rep, keypoint_rep)) / self.args.margin

        return similarity
