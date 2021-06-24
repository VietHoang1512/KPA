import torch
import torch.nn.functional as F

from src.backbone.base_model import BaseModel
from src.losses import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    HardNegativeTripletLoss,
    HardPositiveTripletLoss,
    OnlineContrastiveLoss,
    TripletLoss,
)
from src.mixed.model_argument import MixedModelArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class MixedModel(BaseModel):
    def __init__(self, args: MixedModelArguments):
        """
        Mixed loss model.

        Args:
            args (MixedModelArguments): Mixed Model Argument
        """
        super().__init__(args)

        if self.args.sample_selection == "all":
            self.triplet_loss = TripletLoss(self.distance_metric, triplet_margin=self.args.triplet_margin)
        elif self.args.sample_selection == "pos":
            self.triplet_loss = HardPositiveTripletLoss(self.distance_metric, triplet_margin=self.args.triplet_margin)
        elif self.args.sample_selection == "neg":
            self.triplet_loss = HardNegativeTripletLoss(self.distance_metric, triplet_margin=self.args.triplet_margin)
        elif self.args.sample_selection == "both":
            self.hard_positive_triplet_loss = HardPositiveTripletLoss(
                self.distance_metric, triplet_margin=self.args.triplet_margin
            )
            self.hard_negative_triplet_loss = HardNegativeTripletLoss(
                self.distance_metric, triplet_margin=self.args.triplet_margin
            )
            self.triplet_loss = lambda rep_anchor, rep_pos, rep_neg: self.hard_positive_triplet_loss(
                rep_anchor, rep_pos, rep_neg
            ) + self.hard_negative_triplet_loss(rep_anchor, rep_pos, rep_neg)
        else:
            raise NotImplementedError(
                f"Sample selection strategy {self.args.sample_selection} is not implemented yet. Must be `all`, `pos`, `neg` or `both` "
            )

        if self.args.loss_fct == "constrastive":
            self.pair_loss = ContrastiveLoss(distance_metric=self.distance_metric, margin=args.pair_margin)
        elif self.args.loss_fct == "online-constrastive":
            self.pair_loss = OnlineContrastiveLoss(distance_metric=self.distance_metric, margin=args.pair_margin)
        elif self.args.loss_fct == "consine":
            # FIXME
            logger.warning(f"Provided {self.distance_metric} would be ignored if using cosine similarity loss")
            self.pair_loss = CosineSimilarityLoss()

        else:
            raise NotImplementedError(
                f"Loss function {self.args.loss_fct} is not implemented yet. Must be `constrastive` or `consine`"
            )
        logger.warning(f"Using {self.args.loss_fct} loss function")

    def forward(
        self,
        topic_input_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        topic_token_type_ids: torch.Tensor,
        pos_key_point_input_ids: torch.Tensor,
        pos_key_point_attention_mask: torch.Tensor,
        pos_key_point_token_type_ids: torch.Tensor,
        neg_key_point_input_ids: torch.Tensor,
        neg_key_point_attention_mask: torch.Tensor,
        neg_key_point_token_type_ids: torch.Tensor,
        argument_input_ids: torch.Tensor,
        argument_attention_mask: torch.Tensor,
        argument_token_type_ids: torch.Tensor,
        stance: torch.Tensor,
        label: torch.Tensor,
    ):
        stance_rep = self.fc_stance(stance)

        topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)
        argument_bert_output = self._forward_text(argument_input_ids, argument_attention_mask, argument_token_type_ids)

        pos_key_point_bert_output = self._forward_text(
            pos_key_point_input_ids, pos_key_point_attention_mask, pos_key_point_token_type_ids
        )
        neg_key_point_bert_output = self._forward_text(
            neg_key_point_input_ids, neg_key_point_attention_mask, neg_key_point_token_type_ids
        )

        argument_rep = torch.cat([stance_rep, topic_bert_output, argument_bert_output], axis=1)
        argument_rep = self.fc_text(argument_rep)

        pos_keypoint_rep = torch.cat([stance_rep, topic_bert_output, pos_key_point_bert_output], axis=1)
        pos_keypoint_rep = self.fc_text(pos_keypoint_rep)

        neg_keypoint_rep = torch.cat([stance_rep, topic_bert_output, neg_key_point_bert_output], axis=1)
        neg_keypoint_rep = self.fc_text(neg_keypoint_rep)

        if self.args.normalize:
            argument_rep = F.normalize(argument_rep, p=2, dim=1)
            pos_keypoint_rep = F.normalize(pos_keypoint_rep, p=2, dim=1)
            neg_keypoint_rep = F.normalize(neg_keypoint_rep, p=2, dim=1)

        if self.training:
            pos_idx = label == 1
            neg_idx = label == 0
            pair_idx = label == 2
            pos_loss = 0
            neg_loss = 0
            triplet_loss = 0

            if pos_idx.sum():
                pos_loss = self.pair_loss(
                    output1=argument_rep[pos_idx], output2=pos_keypoint_rep[pos_idx], target=label[pos_idx]
                )
            if neg_idx.sum():
                neg_loss = self.pair_loss(
                    output1=argument_rep[neg_idx], output2=neg_keypoint_rep[neg_idx], target=label[neg_idx]
                )
            if pair_idx.sum():
                triplet_loss = self.triplet_loss(
                    rep_anchor=argument_rep[pair_idx],
                    rep_pos=pos_keypoint_rep[pair_idx],
                    rep_neg=neg_keypoint_rep[pair_idx],
                )

            loss = pos_loss + neg_loss + triplet_loss
            return loss
        similarity = (
            self.args.pair_margin - self.distance_metric(argument_rep, pos_keypoint_rep)
        ) / self.args.pair_margin
        return similarity
