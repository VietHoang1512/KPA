import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.baselines.model_argument import ModelArguments
from src.losses import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    HardNegativeTripletLoss,
    HardPositiveTripletLoss,
    OnlineContrastiveLoss,
    TripletLoss,
)
from src.train_utils.distance import SiameseDistanceMetric
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class MixedModel(nn.Module):
    def __init__(self, args: ModelArguments):
        """
        Simple Bert Siamese Model.

        Args:
            args (ModelArguments): Bert Model Argument
        """
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(
            self.args.model_name_or_path if self.args.model_name_or_path else self.args.model_name,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            from_tf=False,
            output_hidden_states=True,
            return_dict=True,
        )
        self.bert_model = AutoModel.from_pretrained(
            self.args.model_name_or_path if self.args.model_name_or_path else self.args.model_name, config=self.config
        )
        self.model_type = type(self.bert_model).__name__.replace("Model", "").lower()
        self.n_hiddens = self.args.n_hiddens
        self.bert_drop = nn.Dropout(self.args.drop_rate)

        # FIXME
        self.fc_text = nn.Linear(
            self.args.stance_dim + 2 * self.config.hidden_size * max(self.n_hiddens, 1), self.args.text_dim
        )
        self.fc_stance = nn.Linear(1, self.args.stance_dim)

        if self.args.distance == "euclidean":
            self.distance_metric = SiameseDistanceMetric.EUCLIDEAN
        elif self.args.distance == "manhattan":
            self.distance_metric = SiameseDistanceMetric.MANHATTAN
        elif self.args.distance == "cosine":
            self.distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        else:
            raise NotImplementedError(
                f"Embedding similarity function {self.args.distance} is not implemented yet. Must be `euclidean`, `manhattan` or `consine`"
            )

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

        logger.warning(f"Using {self.args.distance} distance function")
        logger.warning(f"Using {self.args.loss_fct} loss function")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward_text(self, input_ids, attention_mask, token_type_ids):
        if self.model_type in ["distilbert", "electra", "bart", "xlm", "xlnet", "camembert", "longformer"]:
            output = self.bert_model(input_ids, attention_mask=attention_mask)
        else:
            output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.n_hiddens > 0:
            hidden_states_key = "hidden_states"
            if self.model_type == "bart":
                hidden_states_key = "decoder_hidden_states"

            if self.model_type == "xlnet" and (not self.training):
                # FIXME: XLnet behaves differenctly between train-eval
                output = torch.cat(
                    [torch.transpose(output[hidden_states_key][-i], 0, 1)[:, 0, :] for i in range(self.n_hiddens)],
                    axis=-1,
                )
            else:
                output = torch.cat([output[hidden_states_key][-i][:, 0, :] for i in range(self.n_hiddens)], axis=-1)
        else:
            output = output["last_hidden_state"][:, 0, :]
        # output = self.text_norm(output)
        output = self.bert_drop(output)
        return output

    def forward(
        self,
        topic_input_ids,
        topic_attention_mask,
        topic_token_type_ids,
        pos_key_point_input_ids,
        pos_key_point_attention_mask,
        pos_key_point_token_type_ids,
        neg_key_point_input_ids,
        neg_key_point_attention_mask,
        neg_key_point_token_type_ids,
        argument_input_ids,
        argument_attention_mask,
        argument_token_type_ids,
        stance,
        label,
    ):
        stance_rep = self.fc_stance(stance)
        # stance_rep = self.stance_norm(stance_rep)

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

        argument_rep = F.normalize(argument_rep, p=2, dim=1)
        pos_keypoint_rep = F.normalize(pos_keypoint_rep, p=2, dim=1)
        neg_keypoint_rep = F.normalize(neg_keypoint_rep, p=2, dim=1)

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
        similarity = (
            self.args.pair_margin - self.distance_metric(argument_rep, pos_keypoint_rep)
        ) / self.args.pair_margin
        return loss, similarity
