from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners

from qs_kpa.backbone.base_model import BaseModel
from qs_kpa.pseudo_label.model_argument import PseudoLabelModelArguments
from qs_kpa.utils.logging import custom_logger

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

            topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids).repeat(
                n_statements, 1
            )

            statement_bert_output = self._forward_text(
                statements_encoded[0][:, 0], statements_encoded[0][:, 1], statements_encoded[0][:, 2]
            )
            if not self.args.stance_free:
                stance_rep = self.fc_stance(stance).repeat(n_statements, 1)
                statement_rep = torch.cat([stance_rep, topic_bert_output, statement_bert_output], axis=1)
            else:
                statement_rep = torch.cat([topic_bert_output, statement_bert_output], axis=1)

            statement_rep = self.fc_text(statement_rep)
            if self.args.normalize:
                statement_rep = F.normalize(statement_rep, p=2, dim=1)

            loss = self.criterion(statement_rep, label[0])

            return loss
        else:

            topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)
            key_point_bert_output = self._forward_text(
                key_point_input_ids, key_point_attention_mask, key_point_token_type_ids
            )
            argument_bert_output = self._forward_text(
                argument_input_ids, argument_attention_mask, argument_token_type_ids
            )
            if not self.args.stance_free:
                stance_rep = self.fc_stance(stance)
                argument_rep = torch.cat([stance_rep, topic_bert_output, argument_bert_output], axis=1)
                keypoint_rep = torch.cat([stance_rep, topic_bert_output, key_point_bert_output], axis=1)
            else:
                argument_rep = torch.cat([topic_bert_output, argument_bert_output], axis=1)
                keypoint_rep = torch.cat([topic_bert_output, key_point_bert_output], axis=1)

            argument_rep = self.fc_text(argument_rep)
            keypoint_rep = self.fc_text(keypoint_rep)
            if self.args.normalize:
                argument_rep = F.normalize(argument_rep, p=2, dim=1)
                keypoint_rep = F.normalize(keypoint_rep, p=2, dim=1)
            similarity = (self.args.margin - self.distance_metric(argument_rep, keypoint_rep)) / self.args.margin

        return similarity

    def get_embeddings(
        self,
        stance: torch.Tensor,
        topic_input_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        topic_token_type_ids: torch.Tensor,
        statement_input_ids: torch.Tensor,
        statement_attention_mask: torch.Tensor,
        statement_token_type_ids: torch.Tensor,
    ):

        topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)
        statement_bert_output = self._forward_text(
            statement_input_ids, statement_attention_mask, statement_token_type_ids
        )
        if not self.args.stance_free:
            stance_rep = self.fc_stance(stance)
            statement_rep = torch.cat([stance_rep, topic_bert_output, statement_bert_output], axis=1)
        else:
            statement_rep = torch.cat([topic_bert_output, statement_bert_output], axis=1)

        statement_rep = self.fc_text(statement_rep)

        if self.args.normalize:
            statement_rep = F.normalize(statement_rep, p=2, dim=1)

        return statement_rep

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += f"\tBackbone: {self.args.model_name_or_path if self.args.model_name_or_path else self.args.model_name}\n"
        s += f"\tNumber of hidden state: {self.args.n_hiddens}\n"
        s += f"\tDropout rate: {self.args.drop_rate}\n"
        s += f"\tUse batch normalization: {self.args.batch_norm}\n"
        s += f"\tHidden representation dimension used for encoding stance: {self.args.stance_dim}\n"
        s += f"\tHidden representation dimension used for encoding text: {self.args.text_dim}\n"
        s += f"\tDistance metric: {self.args.distance}\n"
        s += f"\tNormalize embedding: {self.args.normalize}\n"
        s += f"\tMargin: {self.args.margin}"
        return s
