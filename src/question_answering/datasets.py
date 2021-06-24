from typing import Dict, List, Tuple

import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

from src.backbone.base_dataset import BaseDataset
from src.question_answering.data_argument import QADataArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class QADataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        arg_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: QADataArguments,
    ):
        """
        Bert Keypoint Argument Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            arg_df (pd.DataFrame): DataFrame for all arguments (Used for inference)
            labels_df (pd.DataFrame): DataFrame for labels (Used for inference)
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (QADataArguments): Data Argument
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

        self.max_topic_length = self.args.max_topic_length
        self.max_statement_length = self.args.max_statement_length

        self.sequence_pair_added_tokens = self.tokenizer.model_max_length - self.tokenizer.max_len_sentences_pair
        self.truncation = (
            TruncationStrategy.ONLY_SECOND.value
            if self.tokenizer.padding_side == "right"
            else TruncationStrategy.ONLY_FIRST.value
        )

    def __len__(self):
        """Denotes the number of examples per epoch."""
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, List]:
        """Generate one batch of data."""
        topic = self.topic[idx]
        argument = self.argument[idx]
        key_point = self.key_point[idx]

        encoded_key_point, encoded_argument = self._process(
            topic=topic,
            argument=argument,
            key_point=key_point,
            max_topic_length=self.max_topic_length,
            max_statement_length=self.max_statement_length,
        )

        stance = torch.tensor([self.stance[idx]], dtype=torch.float)

        sample = {
            "key_point_input_ids": torch.tensor(encoded_key_point["input_ids"], dtype=torch.long),
            "key_point_attention_mask": torch.tensor(encoded_key_point["attention_mask"], dtype=torch.long),
            "key_point_token_type_ids": torch.tensor(encoded_key_point["token_type_ids"], dtype=torch.long),
            "argument_input_ids": torch.tensor(encoded_argument["input_ids"], dtype=torch.long),
            "argument_attention_mask": torch.tensor(encoded_argument["attention_mask"], dtype=torch.long),
            "argument_token_type_ids": torch.tensor(encoded_argument["token_type_ids"], dtype=torch.long),
            "stance": stance,
            "label": torch.tensor(self.label[idx], dtype=torch.float),
        }

        return sample

    def _process(
        self, topic: str, argument: str, key_point: str, max_topic_length: int, max_statement_length: int
    ) -> Tuple[Dict[str, List], Dict[str, List]]:
        if self.tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            topic_tokens = self.tokenizer.tokenize(topic, add_prefix_space=True)
        else:
            topic_tokens = self.tokenizer.tokenize(topic)

        key_point_tokens = self.tokenizer.encode(
            key_point, add_special_tokens=False, truncation=True, max_length=max_statement_length
        )
        argument_tokens = self.tokenizer.encode(
            argument, add_special_tokens=False, truncation=True, max_length=max_statement_length
        )
        # Define the side we want to truncate / pad and the text/pair sorting
        # TODO: avoid shallow copy
        if self.tokenizer.padding_side == "right":
            key_point_texts = key_point_tokens
            argument_texts = argument_tokens
            key_point_pairs = topic_tokens
            argument_pairs = topic_tokens

        else:
            key_point_texts = topic_tokens
            argument_texts = topic_tokens
            key_point_pairs = key_point_tokens
            argument_pairs = argument_tokens

        max_length = max_topic_length + max_statement_length + self.sequence_pair_added_tokens

        encoded_key_point = self.tokenizer.encode_plus(
            key_point_texts,
            key_point_pairs,
            truncation=self.truncation,
            padding="max_length",
            max_length=max_length,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
        )

        encoded_argument = self.tokenizer.encode_plus(
            argument_texts,
            argument_pairs,
            truncation=self.truncation,
            padding="max_length",
            max_length=max_length,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
        )
        if len(encoded_key_point["overflowing_tokens"]) > 0 or len(encoded_argument["overflowing_tokens"]) > 0:
            logger.warning(f"String is truncated with maximum length {max_length}")

        return encoded_key_point, encoded_argument
