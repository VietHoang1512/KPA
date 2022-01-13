import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from qs_kpa.backbone.base_arguments import BaseModelArguments
from qs_kpa.train_utils.distance import SiameseDistanceMetric
from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


class BaseModel(nn.Module):
    def __init__(self, args: BaseModelArguments):
        super().__init__()

        if args.stance_free:
            args.stance_dim = 0
        else:
            self.fc_stance = nn.Linear(1, args.stance_dim)

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

        fusion_dim = self.args.stance_dim + 2 * self.config.hidden_size * max(self.n_hiddens, 1)
        if self.args.batch_norm:
            self.fc_text = nn.Sequential(nn.BatchNorm1d(fusion_dim), nn.Linear(fusion_dim, args.text_dim))
        else:
            self.fc_text = nn.Linear(fusion_dim, args.text_dim)

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

    def _forward_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):

        if self.model_type in ["t5", "distilbert", "electra", "bart", "xlm", "xlnet", "camembert", "longformer"]:
            output = self.bert_model(input_ids, attention_mask=attention_mask)
        else:
            output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.n_hiddens > 1:
            hidden_states_key = "hidden_states"
            if self.model_type == "bart":
                hidden_states_key = "decoder_hidden_states"

            # if self.model_type == "xlnet" and (not self.training):
            #     # FIXME: XLnet behaves differenctly between train-eval
            #     output = torch.cat(
            #         [torch.transpose(output[hidden_states_key][-i], 0, 1)[:, 0, :] for i in range(self.n_hiddens)],
            #         axis=-1,
            #     )
            # else:
            output = torch.cat([output[hidden_states_key][-i][:, 0, :] for i in range(self.n_hiddens)], axis=-1)
        elif self.n_hiddens == 0:
            output = output["pooler_output"]
        elif self.n_hiddens == 1:
            output = output["last_hidden_state"][:, 0, :]
        else:
            token_embeddings = output = output["last_hidden_state"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        output = self.bert_drop(output)
        return output
