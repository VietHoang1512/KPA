from typing import Optional

import torch
import torch.nn.functional as F

from src.backbone.base_model import BaseModel
from src.losses import (
    HardNegativeTripletLoss,
    HardPositiveTripletLoss,
    TripletLoss,
)
from src.triplet.model_argument import TripletModelArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class TripletModel(BaseModel):
    def __init__(self, args: TripletModelArguments):
        """
        Pseudo Label Model.

        Args:
            args (TripletModelArguments): Triplet Model Argument
        """
        super().__init__(args)

        if self.args.sample_selection == "all":
            self.triplet_loss = TripletLoss(self.distance_metric, triplet_margin=self.args.margin)
        elif self.args.sample_selection == "pos":
            self.triplet_loss = HardPositiveTripletLoss(self.distance_metric, triplet_margin=self.args.margin)
        elif self.args.sample_selection == "neg":
            self.triplet_loss = HardNegativeTripletLoss(self.distance_metric, triplet_margin=self.args.margin)
        elif self.args.sample_selection == "both":
            self.hard_positive_triplet_loss = HardPositiveTripletLoss(
                self.distance_metric, triplet_margin=self.args.margin
            )
            self.hard_negative_triplet_loss = HardNegativeTripletLoss(
                self.distance_metric, triplet_margin=self.args.margin
            )
            self.triplet_loss = lambda rep_anchor, rep_pos, rep_neg: self.hard_positive_triplet_loss(
                rep_anchor, rep_pos, rep_neg
            ) + self.hard_negative_triplet_loss(rep_anchor, rep_pos, rep_neg)
        else:
            raise NotImplementedError(
                f"Sample selection strategy {self.args.sample_selection} is not implemented yet. Must be `all`, `pos`, `neg` or `both` "
            )

    def forward(
        self,
        topic_input_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        topic_token_type_ids: torch.Tensor,
        stance: torch.Tensor,
        key_point_input_ids: torch.Tensor,
        key_point_attention_mask: torch.Tensor,
        key_point_token_type_ids: torch.Tensor,
        pos_argument_input_ids: torch.Tensor,
        pos_argument_attention_mask: torch.Tensor,
        pos_argument_token_type_ids: torch.Tensor,
        neg_argument_input_ids: Optional[torch.Tensor],
        neg_argument_attention_mask: Optional[torch.Tensor],
        neg_argument_token_type_ids: Optional[torch.Tensor],
    ):
        stance_rep = self.fc_stance(stance)
        topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)

        key_point_bert_output = self._forward_text(
            key_point_input_ids, key_point_attention_mask, key_point_token_type_ids
        )

        pos_argument_bert_output = self._forward_text(
            pos_argument_input_ids, pos_argument_attention_mask, pos_argument_token_type_ids
        )

        neg_argument_bert_output = self._forward_text(
            neg_argument_input_ids, neg_argument_attention_mask, neg_argument_token_type_ids
        )

        pos_argument_rep = torch.cat([stance_rep, topic_bert_output, pos_argument_bert_output], axis=1)
        pos_argument_rep = self.fc_text(pos_argument_rep)

        neg_argument_rep = torch.cat([stance_rep, topic_bert_output, neg_argument_bert_output], axis=1)
        neg_argument_rep = self.fc_text(neg_argument_rep)

        keypoint_rep = torch.cat([stance_rep, topic_bert_output, key_point_bert_output], axis=1)
        keypoint_rep = self.fc_text(keypoint_rep)

        if self.args.normalize:
            keypoint_rep = F.normalize(keypoint_rep, p=2, dim=1)
            pos_argument_rep = F.normalize(pos_argument_rep, p=2, dim=1)
            neg_argument_rep = F.normalize(neg_argument_rep, p=2, dim=1)

        if self.training:

            loss = self.triplet_loss(
                rep_anchor=keypoint_rep,
                rep_pos=pos_argument_rep,
                rep_neg=neg_argument_rep,
            )
            return loss
        similarity = (self.args.margin - self.distance_metric(keypoint_rep, pos_argument_rep)) / self.args.margin
        return similarity
