import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners
from transformers import AutoConfig, AutoModel

from src.pseudo_label.model_argument import ModelArguments
from src.train_utils.distance import SiameseDistanceMetric
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class PseudoLabelModel(nn.Module):
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
        if self.args.distance == "cosine":
            self.distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
            self.distance = distances.CosineSimilarity()
        elif self.args.distance == "euclidean":
            self.distance_metric = SiameseDistanceMetric.EUCLIDEAN
            self.distance = distances.LpDistance()
        else:
            raise NotImplementedError(
                f"Embedding similarity function {self.args.distance} is not implemented yet. Must be `consine` or `euclidean`"
            )
        self.loss_func = losses.TripletMarginLoss(margin=self.args.margin, distance=self.distance)
        self.mining_func = miners.TripletMarginMiner(
            margin=self.args.margin, distance=self.distance, type_of_triplets="semihard"
        )

    def criterion(self, embeddings, labels):
        # indices_tuple = self.mining_func(embeddings, labels)
        # loss = self.loss_func(embeddings, labels, indices_tuple)
        loss = self.loss_func(embeddings, labels)
        return loss

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
        statements_encoded,
        stance,
        label,
    ):

        n_statements = statements_encoded[0].shape[0]

        stance_rep = self.fc_stance(stance).repeat(n_statements, 1)
        topic_bert_output = self._forward_text(topic_input_ids, topic_attention_mask, topic_token_type_ids).repeat(
            n_statements, 1
        )

        statement_bert_output = self._forward_text(
            statements_encoded[0][:, 0], statements_encoded[0][:, 1], statements_encoded[0][:, 2]
        )
        statement_rep = torch.cat([stance_rep, topic_bert_output, statement_bert_output], axis=1)
        statement_rep = self.fc_text(statement_rep)
        statements_rep = F.normalize(statement_rep, p=2, dim=1)

        loss = self.criterion(statement_rep, label[0])

        similarity = (
            self.args.margin - self.distance_metric(statement_rep[0].view(1, -1), statements_rep[1].view(1, -1))
        ) / self.args.margin

        return loss, similarity
