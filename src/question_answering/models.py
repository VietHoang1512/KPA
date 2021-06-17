import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.losses import (
    ContrastiveLoss,
    CosineSimilarityLoss,
    OnlineContrastiveLoss,
)
from src.question_answering.model_argument import ModelArguments
from src.train_utils.distance import SiameseDistanceMetric
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class BertQAModel(nn.Module):
    def __init__(self, args: ModelArguments):
        """
        Bert Question Answering like Model.

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

        self.fc_text = nn.Linear(
            self.args.stance_dim + self.config.hidden_size * max(self.n_hiddens, 1), self.args.text_dim
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

        output = self.bert_drop(output)
        return output

    def forward(
        self,
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

        key_point_bert_output = self._forward_text(
            key_point_input_ids, key_point_attention_mask, key_point_token_type_ids
        )
        argument_bert_output = self._forward_text(argument_input_ids, argument_attention_mask, argument_token_type_ids)

        argument_rep = torch.cat([stance_rep, argument_bert_output], axis=1)
        argument_rep = self.fc_text(argument_rep)
        keypoint_rep = torch.cat([stance_rep, key_point_bert_output], axis=1)
        keypoint_rep = self.fc_text(keypoint_rep)

        argument_rep = F.normalize(argument_rep, p=2, dim=1)
        keypoint_rep = F.normalize(keypoint_rep, p=2, dim=1)

        loss = self.criterion(output1=argument_rep, output2=keypoint_rep, target=label)

        similarity = F.relu(self.args.margin - self.distance_metric(argument_rep, keypoint_rep)) / self.args.margin
        return loss, similarity
