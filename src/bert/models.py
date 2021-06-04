import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.bert.model_argument import ModelArguments


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):

        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class BertKPAModel(nn.Module):
    def __init__(self, args: ModelArguments):

        super().__init__()

        config = AutoConfig.from_pretrained(
            args.model_name,
            from_tf=False,
            output_hidden_states=True,
        )
        self.bert_model = AutoModel.from_pretrained(args.model_name, config=config)
        self.n_hiddens = args.n_hiddens
        self.bert_drop = nn.Dropout(args.drop_rate)

        self.fc_argument = nn.Linear(config.hidden_size * self.n_hiddens, args.text_dim)
        self.fc_stance = nn.Linear(1, args.stance_dim)
        self.fc_keypoint = nn.Linear(args.stance_dim + 2 * config.hidden_size * self.n_hiddens, args.text_dim)

        self.criterion = ContrastiveLoss()

    def _forward_text(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = torch.mean(torch.cat([output[2][-i] for i in range(self.n_hiddens)], axis=-1), axis=1)
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

        argument_rep = self.fc_argument(argument_bert_output)
        keypoint_rep = torch.cat([stance_rep, topic_bert_output, key_point_bert_output], axis=1)
        keypoint_rep = self.fc_keypoint(keypoint_rep)

        argument_rep = F.normalize(argument_rep, p=2, dim=1)
        keypoint_rep = F.normalize(keypoint_rep, p=2, dim=1)
        euclidean_distance = F.pairwise_distance(argument_rep, keypoint_rep)

        loss = self.criterion(euclidean_distance, label)
        return loss, euclidean_distance
