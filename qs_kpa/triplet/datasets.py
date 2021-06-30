import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from qs_kpa.backbone.base_dataset import BaseDataset
from qs_kpa.triplet.data_argument import TripletDataArguments
from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


class TripletTrainDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: TripletDataArguments,
    ):
        """
        Triplet Bert Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (TripletDataArguments): Triplet Data Argument
        """
        super().__init__(tokenizer, args)
        self.data = self._process_data(df.copy())

    def __len__(self):
        """Denotes the number of examples per epoch."""
        return len(self.data)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        datum: Dict = self.data[idx]

        stance = torch.tensor([datum["stance"]], dtype=torch.float)
        topic = datum["topic"]
        key_point = datum["key_point"]
        pos_argument = random.choice(datum["pos"])
        neg_argument = random.choice(datum["neg"])
        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(
            text=topic, max_len=self.args.max_len
        )
        key_point_input_ids, key_point_attention_mask, key_point_token_type_ids = self._tokenize(
            text=key_point, max_len=self.args.max_len
        )
        pos_argument_input_ids, pos_argument_attention_mask, pos_argument_token_type_ids = self._tokenize(
            text=pos_argument, max_len=self.args.argument_max_len
        )

        neg_argument_input_ids, neg_argument_attention_mask, neg_argument_token_type_ids = self._tokenize(
            text=neg_argument, max_len=self.args.argument_max_len
        )

        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "key_point_input_ids": key_point_input_ids,
            "key_point_attention_mask": key_point_attention_mask,
            "key_point_token_type_ids": key_point_token_type_ids,
            "pos_argument_input_ids": pos_argument_input_ids,
            "pos_argument_attention_mask": pos_argument_attention_mask,
            "pos_argument_token_type_ids": pos_argument_token_type_ids,
            "neg_argument_input_ids": neg_argument_input_ids,
            "neg_argument_attention_mask": neg_argument_attention_mask,
            "neg_argument_token_type_ids": neg_argument_token_type_ids,
            "stance": stance,
        }

        return sample

    def _process_data(self, df: pd.DataFrame) -> List[Dict]:

        data = []
        cnt_neg = []
        cnt_pos = []

        for _, key_point_id_df in df.groupby(["key_point_id"]):
            key_point_id_dict = {"neg": [], "pos": []}
            key_point_id_dict.update(key_point_id_df.iloc[0].to_dict())
            for _, row in key_point_id_df.iterrows():
                if row["label"] == 1:
                    key_point_id_dict["pos"].append(row["argument"])
                else:
                    key_point_id_dict["neg"].append(row["argument"])
            n_pos = len(key_point_id_dict["pos"])
            n_neg = len(key_point_id_dict["neg"])
            cnt_neg.append(n_neg)
            cnt_pos.append(n_pos)
            if n_neg * n_pos:
                data.append(key_point_id_dict)
        logger.warning(
            f"No. negative arguments Mean: {np.mean(cnt_neg):.2f} \u00B1 {np.std(cnt_neg):.2f}  Max: {np.max(cnt_neg)}  Median: {np.median(cnt_neg):.2f}"
        )
        logger.warning(
            f"No. postive arguments Mean: {np.mean(cnt_pos):.2f} \u00B1 {np.std(cnt_pos):.2f}  Max: {np.max(cnt_pos)}  Median: {np.median(cnt_pos):.2f}"
        )
        return data


class TripletInferenceDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        arg_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: TripletDataArguments,
    ):
        """
        Triplet Inference Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            arg_df (pd.DataFrame): DataFrame for all arguments (Used for inference)
            labels_df (pd.DataFrame): DataFrame for labels (Used for inference)
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (TripletDataArguments): Triplet Data Argument
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
            "pos_argument_input_ids": argument_input_ids,
            "pos_argument_attention_mask": argument_attention_mask,
            "pos_argument_token_type_ids": argument_token_type_ids,
            "neg_argument_input_ids": argument_input_ids,
            "neg_argument_attention_mask": argument_attention_mask,
            "neg_argument_token_type_ids": argument_token_type_ids,
            "key_point_input_ids": key_point_input_ids,
            "key_point_attention_mask": key_point_attention_mask,
            "key_point_token_type_ids": key_point_token_type_ids,
            "stance": stance,
        }

        return sample
