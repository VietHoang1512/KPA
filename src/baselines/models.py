import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from src.backbone.base_model import BaseModel
from src.baselines.model_argument import BaselineModelArguments
from src.losses import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    OnlineContrastiveLoss,
)
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class BaselineBertModel(BaseModel):
    def __init__(self, args: BaselineModelArguments):
        """
        Simple Bert Siamese Model.

        Args:
            args (ModelArguments): Bert Model Argument
        """
        super().__init__(args)

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
        topic_input_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        topic_token_type_ids: torch.Tensor,
        key_point_input_ids: torch.Tensor,
        key_point_attention_mask: torch.Tensor,
        key_point_token_type_ids: torch.Tensor,
        argument_input_ids: torch.Tensor,
        argument_attention_mask: torch.Tensor,
        argument_token_type_ids: torch.Tensor,
        stance: torch.Tensor,
        label: torch.Tensor,
    ):
        stance_rep = self.fc_stance(stance)

        topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)
        key_point_bert_output = self._forward_text(
            key_point_input_ids, key_point_attention_mask, key_point_token_type_ids
        )
        argument_bert_output = self._forward_text(argument_input_ids, argument_attention_mask, argument_token_type_ids)

        argument_rep = torch.cat([stance_rep, topic_bert_output, argument_bert_output], axis=1)
        argument_rep = self.fc_text(argument_rep)
        keypoint_rep = torch.cat([stance_rep, topic_bert_output, key_point_bert_output], axis=1)
        keypoint_rep = self.fc_text(keypoint_rep)

        argument_rep = F.normalize(argument_rep, p=2, dim=1)
        keypoint_rep = F.normalize(keypoint_rep, p=2, dim=1)

        if self.training:
            loss = self.criterion(output1=argument_rep, output2=keypoint_rep, target=label)
            return loss

        similarity = (self.args.margin - self.distance_metric(argument_rep, keypoint_rep)) / self.args.margin
        return similarity


class BertKPAClassificationModel(BaseModel):
    def __init__(self, args: BaselineModelArguments):
        """
        Simple Bert Classification Model.

        Args:
            args (BaselineModelArguments): Bert Model Argument
        """
        super().__init__(args)

        config = AutoConfig.from_pretrained(
            args.model_name,
            from_tf=False,
            output_hidden_states=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(args.stance_dim + 3 * config.hidden_size * self.n_hiddens, 1), nn.ReLU(inplace=True)
        )
        self.fc_stance = nn.Sequential(nn.Linear(1, args.stance_dim), nn.ReLU(inplace=True))

    @classmethod
    def criterion(self, preds, label):
        label = label.view(-1, 1)
        return nn.BCELoss()(preds, label)

    def forward(
        self,
        topic_input_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        topic_token_type_ids: torch.Tensor,
        key_point_input_ids: torch.Tensor,
        key_point_attention_mask: torch.Tensor,
        key_point_token_type_ids: torch.Tensor,
        argument_input_ids: torch.Tensor,
        argument_attention_mask: torch.Tensor,
        argument_token_type_ids: torch.Tensor,
        stance: torch.Tensor,
        label: torch.Tensor,
    ):
        stance_ouput = self.fc_stance(stance)
        topic_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)
        key_point_output = self._forward_text(key_point_input_ids, key_point_attention_mask, key_point_token_type_ids)
        argument_output = self._forward_text(argument_input_ids, argument_attention_mask, argument_token_type_ids)

        output = torch.cat([stance_ouput, topic_output, key_point_output, argument_output], axis=1)

        output = self.fc(output)
        prob = torch.sigmoid(output)

        loss = self.criterion(prob, label)
        return loss, prob
