import random
from typing import Dict, List

import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from src.backbone.base_dataset import BaseDataset
from src.mixed.data_argument import MixedDataArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class MixedTrainDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: MixedDataArguments,
    ):
        """
        Mixed Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (MixedDataArguments): Mixed Data Argument
        """
        super().__init__(tokenizer, args)
        df = df.copy()
        self.data = self._process_data(df)

    def __len__(self):
        """Denotes the number of examples per epoch."""
        return len(self.data)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        datum: Dict = self.data[idx]

        stance = torch.tensor([datum["stance"]], dtype=torch.float)
        topic = datum["topic"]
        argument = datum["argument"]

        label = 2  # this sample has both negative and positive key point
        if len(datum[0]):
            neg_key_point = random.choice(datum[0])
        else:
            label = 1
            neg_key_point = "this sample doesnt contain negative keypoint"

        if len(datum[1]):
            pos_key_point = random.choice(datum[1])
        else:
            label = 0
            pos_key_point = "this sample doesnt contain positive keypoint"

        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(
            text=topic, max_len=self.args.max_len
        )

        argument_input_ids, argument_attention_mask, argument_token_type_ids = self._tokenize(
            text=argument, max_len=self.args.argument_max_len
        )

        pos_key_point_input_ids, pos_key_point_attention_mask, pos_key_point_token_type_ids = self._tokenize(
            text=pos_key_point, max_len=self.args.max_len
        )
        neg_key_point_input_ids, neg_key_point_attention_mask, neg_key_point_token_type_ids = self._tokenize(
            text=neg_key_point, max_len=self.args.max_len
        )

        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "pos_key_point_input_ids": pos_key_point_input_ids,
            "pos_key_point_attention_mask": pos_key_point_attention_mask,
            "pos_key_point_token_type_ids": pos_key_point_token_type_ids,
            "neg_key_point_input_ids": neg_key_point_input_ids,
            "neg_key_point_attention_mask": neg_key_point_attention_mask,
            "neg_key_point_token_type_ids": neg_key_point_token_type_ids,
            "argument_input_ids": argument_input_ids,
            "argument_attention_mask": argument_attention_mask,
            "argument_token_type_ids": argument_token_type_ids,
            "stance": stance,
            "label": torch.tensor(label, dtype=torch.float),
        }

        return sample

    def _process_data(self, df: pd.DataFrame) -> List[Dict]:
        data = []
        cnt_neg = 0
        cnt_pos = 0
        for _, arg_id_df in df.groupby(["arg_id"]):
            arg_id_dict = {0: [], 1: []}
            arg_id_dict.update(arg_id_df.iloc[0].to_dict())
            for _, row in arg_id_df.iterrows():
                arg_id_dict[row["label"]].append(row["key_point"])
            if len(arg_id_dict[0]) == 0:
                cnt_neg += 1
            if len(arg_id_dict[1]) == 0:
                cnt_pos += 1
            data.append(arg_id_dict)
        logger.warning(
            f"There are {cnt_neg} arguments without negative and {cnt_pos} postitive key points in total {len(data)}"
        )
        return data


class MixedInferenceDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        arg_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: MixedDataArguments,
    ):
        """
        Mixed Inference Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            arg_df (pd.DataFrame): DataFrame for all arguments (Used for inference)
            labels_df (pd.DataFrame): DataFrame for labels (Used for inference)
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (MixedDataArguments): Mixed Data Argument
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

        # Duplicate the postive and negative samples
        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "pos_key_point_input_ids": key_point_input_ids,
            "pos_key_point_attention_mask": key_point_attention_mask,
            "pos_key_point_token_type_ids": key_point_token_type_ids,
            "neg_key_point_input_ids": key_point_input_ids,
            "neg_key_point_attention_mask": key_point_attention_mask,
            "neg_key_point_token_type_ids": key_point_token_type_ids,
            "argument_input_ids": argument_input_ids,
            "argument_attention_mask": argument_attention_mask,
            "argument_token_type_ids": argument_token_type_ids,
            "stance": stance,
            "label": torch.tensor(self.label[idx], dtype=torch.float),
        }

        return sample
