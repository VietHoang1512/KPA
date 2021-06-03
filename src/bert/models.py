import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class BertKPAModel(nn.Module):
    def __init__(self, bert_model, n_hiddens, stance_dim=32, drop_rate=0.0, text_dim=32):
        super().__init__()

        self.fc_stance = nn.Linear(1, stance_dim)

        config = AutoConfig.from_pretrained(
            bert_model,
            from_tf=False,
            output_hidden_states=True,
        )
        self.bert_model = AutoModel.from_pretrained(bert_model, config=config)
        self.n_hiddens = n_hiddens
        self.fc_topic = nn.Linear(config.hidden_size * self.n_hiddens, text_dim)
        self.fc_key_point = nn.Linear(config.hidden_size * self.n_hiddens, text_dim)
        self.fc_argument = nn.Linear(config.hidden_size * self.n_hiddens, text_dim)
        self.bert_drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(stance_dim + 3 * text_dim, 1)
        self.criterion = nn.BCELoss()

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

        stance_ouput = F.relu(self.fc_stance(stance))

        topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids)
        topic_output = self.fc_topic(topic_bert_output)

        key_point_bert_output = self._forward_text(
            key_point_input_ids, key_point_attention_mask, key_point_token_type_ids
        )
        key_point_output = self.fc_topic(key_point_bert_output)

        argument_bert_output = self._forward_text(argument_input_ids, argument_attention_mask, argument_token_type_ids)
        argument_output = self.fc_argument(argument_bert_output)

        output = torch.cat([stance_ouput, topic_output, key_point_output, argument_output], axis=1)

        output = self.fc(output)
        prob = torch.sigmoid(output)

        loss = self.criterion(prob, label)
        return loss, prob
