import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from src.backbone.base_dataset import BaseDataset
from src.baselines.data_argument import BaselineDataArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class BaselineBertDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        arg_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: BaselineDataArguments,
    ):
        """
        Baseline Bert Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            arg_df (pd.DataFrame): DataFrame for all arguments (Used for inference)
            labels_df (pd.DataFrame): DataFrame for labels (Used for inference)
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (BaselineDataArguments): Baseline Data Arguments
        """
        super().__init__(tokenizer, args)
        df = df.copy()
        self.df = df
        self.arg_df = arg_df.copy()
        self.labels_df = labels_df.copy()
        self.topic = df["topic"].tolist()
        self.argument = df["argument"].tolist()
        self.key_point = df["key_point"].tolist()
        self.label = df["label"].values
        self.stance = df["stance"].values

    def __len__(self):
        """Denotes the number of examples per epoch."""
        return len(self.df)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        topic = self.topic[idx]
        argument = self.argument[idx]
        key_point = self.key_point[idx]

        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(
            text=topic, max_len=self.args.max_len
        )
        key_point_input_ids, key_point_attention_mask, key_point_token_type_ids = self._tokenize(
            text=key_point, max_len=self.args.max_len
        )
        argument_input_ids, argument_attention_mask, argument_token_type_ids = self._tokenize(
            text=argument, max_len=self.args.argument_max_len
        )
        stance = torch.tensor([self.stance[idx]], dtype=torch.float)

        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "key_point_input_ids": key_point_input_ids,
            "key_point_attention_mask": key_point_attention_mask,
            "key_point_token_type_ids": key_point_token_type_ids,
            "argument_input_ids": argument_input_ids,
            "argument_attention_mask": argument_attention_mask,
            "argument_token_type_ids": argument_token_type_ids,
            "stance": stance,
            "label": torch.tensor(self.label[idx], dtype=torch.float),
        }

        return sample
