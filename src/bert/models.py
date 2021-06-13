import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.bert.distance import SiameseDistanceMetric
from src.bert.losses import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    OnlineContrastiveLoss,
)
from src.bert.model_argument import ModelArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class BertSiameseModel(nn.Module):
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
        )
        self.bert_model = AutoModel.from_pretrained(
            self.args.model_name_or_path if self.args.model_name_or_path else self.args.model_name, config=self.config
        )

        self.n_hiddens = self.args.n_hiddens
        self.bert_drop = nn.Dropout(self.args.drop_rate)

        # self.stance_norm = nn.LayerNorm(args.stance_dim)
        # self.text_norm = nn.LayerNorm(self.config.hidden_size * self.n_hiddens)

        # FIXME
        # self.fc_text = AttentionHead(self.args.stance_dim + 2 * self.config.hidden_size
        #                              * self.n_hiddens, self.args.text_dim, self.args.text_dim)
        self.fc_text = nn.Linear(
            self.args.stance_dim + 2 * self.config.hidden_size * self.n_hiddens, self.args.text_dim
        )
        self.fc_stance = nn.Linear(1, self.args.stance_dim)

        # self.fc_text = nn.Sequential(
        #     nn.Linear(args.stance_dim + 2 * self.config.hidden_size * self.n_hiddens, args.text_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(args.text_dim, args.text_dim),
        # )
        # self.fc_stance = nn.Sequential(nn.Linear(1, args.stance_dim), nn.ReLU(inplace=True), nn.Linear(args.stance_dim, args.stance_dim))

        # self._init_weights(self.text_norm)
        # self._init_weights(self.fc_text)
        # self._init_weights(self.fc_stance)

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
        output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = torch.cat([output[2][-i][:, 0, :] for i in range(self.n_hiddens)], axis=-1)
        # output = self.text_norm(output)
        output = self.bert_drop(output)
        return output

    def forward(
        self,
        topic_input_ids,
        topic_attention_mask,
        topic_token_type_ids,
        key_point_input_ids,
        key_point_attention_mask,
        key_point_token_type_ids,
        argument_input_ids,
        argument_attention_mask,
        argument_token_type_ids,
        stance,
        label,
    ):
        stance_rep = self.fc_stance(stance)
        # stance_rep = self.stance_norm(stance_rep)

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

        loss = self.criterion(output1=argument_rep, output2=keypoint_rep, target=label)

        similarity = (self.args.margin - self.distance_metric(argument_rep, keypoint_rep)) / self.args.margin
        return loss, similarity


class BertKPAClassificationModel(nn.Module):
    def __init__(self, args: ModelArguments):
        """
        Simple Bert Classification Model.

        Args:
            args (ModelArguments): Bert Model Argument
        """
        super().__init__()

        config = AutoConfig.from_pretrained(
            args.model_name,
            from_tf=False,
            output_hidden_states=True,
        )
        self.args = args
        self.bert_model = AutoModel.from_pretrained(args.model_name, config=config)
        self.n_hiddens = args.n_hiddens
        self.bert_drop = nn.Dropout(args.drop_rate)

        self.fc = nn.Sequential(
            nn.Linear(args.stance_dim + 3 * config.hidden_size * self.n_hiddens, 1), nn.ReLU(inplace=True)
        )
        self.fc_stance = nn.Sequential(nn.Linear(1, args.stance_dim), nn.ReLU(inplace=True))

    @classmethod
    def criterion(self, preds, label):
        label = label.view(-1, 1)
        return nn.BCELoss()(preds, label)

    def _forward_text(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = torch.cat([output[2][-i][:, 0, :] for i in range(self.n_hiddens)], axis=-1)
        output = self.bert_drop(output)
        return output

    def forward(
        self,
        topic_input_ids,
        topic_attention_mask,
        topic_token_type_ids,
        key_point_input_ids,
        key_point_attention_mask,
        key_point_token_type_ids,
        argument_input_ids,
        argument_attention_mask,
        argument_token_type_ids,
        stance,
        label,
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


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))

        score = self.V(att)

        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector
