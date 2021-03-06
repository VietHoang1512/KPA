import torch
import torch.nn as nn
import torch.nn.functional as F

from qs_kpa.backbone.base_model import BaseModel
from qs_kpa.losses import ContrastiveLoss, CosineSimilarityLoss, OnlineContrastiveLoss
from qs_kpa.question_answering.model_argument import QAModelArguments
from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


class QABertModel(BaseModel):
    def __init__(self, args: QAModelArguments):
        """
        Bert Question Answering like Model.

        Args:
            args (QAModelArguments): Question Answering Model Argument
        """
        super().__init__(args)

        fusion_dim = self.args.stance_dim + self.config.hidden_size * max(self.n_hiddens, 1)
        if self.args.batch_norm:
            self.fc_text = nn.Sequential(nn.BatchNorm1d(fusion_dim), nn.Linear(fusion_dim, args.text_dim))
        else:
            self.fc_text = nn.Linear(fusion_dim, args.text_dim)

        if self.args.loss_fct == "constrastive":
            self.criterion = ContrastiveLoss(distance_metric=self.distance_metric, margin=args.margin)
        elif self.args.loss_fct == "online-constrastive":
            self.criterion = OnlineContrastiveLoss(distance_metric=self.distance_metric, margin=args.margin)
        elif self.args.loss_fct == "consine":
            # FIXME
            logger.warning(f"Provided {self.distance_metric} would be ignored if using cosine similarity loss")
            self.criterion = CosineSimilarityLoss()

        else:
            raise NotImplementedError(
                f"Loss function {self.args.loss_fct} is not implemented yet. Must be `constrastive` or `consine`"
            )

        logger.warning(f"Using {self.args.loss_fct} loss function")

    def forward(
        self,
        key_point_input_ids: torch.Tensor,
        key_point_attention_mask: torch.Tensor,
        key_point_token_type_ids: torch.Tensor,
        argument_input_ids: torch.Tensor,
        argument_attention_mask: torch.Tensor,
        argument_token_type_ids: torch.Tensor,
        stance: torch.Tensor,
        label: torch.Tensor,
    ):

        argument_rep = self._forward_text(key_point_input_ids, key_point_attention_mask, key_point_token_type_ids)
        keypoint_rep = self._forward_text(argument_input_ids, argument_attention_mask, argument_token_type_ids)

        if not self.args.stance_free:
            stance_rep = self.fc_stance(stance)
            argument_rep = torch.cat([stance_rep, argument_rep], axis=1)
            keypoint_rep = torch.cat([stance_rep, keypoint_rep], axis=1)

        argument_rep = self.fc_text(argument_rep)
        keypoint_rep = self.fc_text(keypoint_rep)

        if self.args.normalize:
            argument_rep = F.normalize(argument_rep, p=2, dim=1)
            keypoint_rep = F.normalize(keypoint_rep, p=2, dim=1)

        if self.training:
            loss = self.criterion(output1=argument_rep, output2=keypoint_rep, target=label)
            return loss

        similarity = (self.args.margin - self.distance_metric(argument_rep, keypoint_rep)) / self.args.margin
        return similarity
