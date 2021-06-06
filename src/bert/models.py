import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.bert.losses import ContrastiveLoss
from src.bert.model_argument import ModelArguments


class BertKPAModel(nn.Module):
    def __init__(self, args: ModelArguments):
        """
        Simple Bert Siamese Model.

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

        self.fc_text = nn.Sequential(
            nn.Linear(args.stance_dim + 2 * config.hidden_size * self.n_hiddens, args.text_dim), nn.ReLU(inplace=True)
        )
        self.fc_stance = nn.Sequential(nn.Linear(1, args.stance_dim), nn.ReLU(inplace=True))

        self.criterion = ContrastiveLoss(margin=args.margin)

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

        loss = self.criterion(argument_rep, keypoint_rep, label)

        similarity = F.relu(self.args.margin - (argument_rep - keypoint_rep).pow(2).sum(1)) / self.args.margin
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

        self.fc_text = nn.Sequential(
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
