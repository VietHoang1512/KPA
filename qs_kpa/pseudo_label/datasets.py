import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from qs_kpa.backbone.base_dataset import BaseDataset
from qs_kpa.pseudo_label.data_argument import PseudoLabelDataArguments
from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


def _extract_id(key_point_id: str) -> float:
    return float(key_point_id.split("_")[-1]) + 1


class PseudoLabelTrainDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: PseudoLabelDataArguments,
    ):
        """
        Bert Keypoint Argument Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (PseudoLabelDataArguments): Data Argument
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

        random.shuffle(datum["pos"])

        temp = list(zip(datum["neg"], datum["neg_class"]))
        random.shuffle(temp)

        try:
            datum["neg"], datum["neg_class"] = zip(*temp)
            datum["neg"] = list(datum["neg"])
            datum["neg_class"] = list(datum["neg_class"])
        except ValueError:
            logger.warning(f'`{datum["key_point"]}` has no negative argument')

        n_pos = min(len(datum["pos"]), self.args.max_pos)
        n_neg = min(len(datum["neg"]), self.args.max_neg)
        n_unknown = min(len(datum["unknown"]), self.args.max_unknown)

        statements = [datum["key_point"]] + datum["pos"][:n_pos] + datum["neg"][:n_neg] + datum["unknown"][:n_unknown]
        label = (
            [datum["class"]] * (n_pos + 1)
            + datum["neg_class"][:n_neg]
            + list(range(self.max_topic, self.max_topic + n_unknown))
        )
        # print(label)
        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(
            text=topic, max_len=self.args.max_len
        )

        statements_encoded = []
        for statement in statements:
            statements_encoded.append(
                torch.stack(self._tokenize(text=statement, max_len=self.args.statement_max_len), axis=0)
            )

        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "statements_encoded": torch.stack(statements_encoded, axis=0),
            "stance": stance,
            "label": torch.tensor(label, dtype=torch.float),
        }

        return sample

    def _process_data(self, df: pd.DataFrame) -> List[Dict]:
        arg2kp = df[df["label"] == 1].set_index("arg_id")["key_point_id"].map(_extract_id).to_dict()
        df["class"] = df["arg_id"].map(arg2kp).fillna(0).astype(int)
        self.max_topic = df["class"].max() + 1
        data = []
        cnt_neg = []
        cnt_pos = []
        cnt_unknown = []
        for key_point_id, key_point_id_df in df.groupby(["key_point_id"]):
            key_point_id_dict = {"neg": [], "pos": [], "neg_class": [], "unknown": []}
            key_point_id_dict.update(key_point_id_df.iloc[0].to_dict())
            key_point_id_dict["class"] = _extract_id(key_point_id)
            for _, row in key_point_id_df.iterrows():
                if row["label"] == 1:
                    key_point_id_dict["pos"].append(row["argument"])
                elif row["class"]:
                    key_point_id_dict["neg_class"].append(row["class"])
                    key_point_id_dict["neg"].append(row["argument"])
                else:
                    key_point_id_dict["unknown"].append(row["argument"])
            cnt_neg.append(len(key_point_id_dict["neg"]))
            cnt_pos.append(len(key_point_id_dict["pos"]))
            cnt_unknown.append(len(key_point_id_dict["unknown"]))

            data.append(key_point_id_dict)
        logger.warning(
            f"No. negative arguments Mean: {np.mean(cnt_neg):.2f} \u00B1 {np.std(cnt_neg):.2f}  Max: {np.max(cnt_neg)}  Median: {np.median(cnt_neg):.2f}"
        )
        logger.warning(
            f"No. postive arguments Mean: {np.mean(cnt_pos):.2f} \u00B1 {np.std(cnt_pos):.2f}  Max: {np.max(cnt_pos)}  Median: {np.median(cnt_pos):.2f}"
        )
        logger.warning(
            f"No. unknown arguments Mean: {np.mean(cnt_unknown):.2f} \u00B1 {np.std(cnt_unknown):.2f}  Max: {np.max(cnt_unknown)}  Median: {np.median(cnt_unknown):.2f}"
        )
        return data


class PseudoLabelInferenceDataset(BaseDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        arg_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: PseudoLabelDataArguments,
    ):
        """
        Bert Keypoint Argument Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            arg_df (pd.DataFrame): DataFrame for all arguments (Used for inference)
            labels_df (pd.DataFrame): DataFrame for labels (Used for inference)
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (PseudoLabelDataArguments): Data Argument
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
        argument_input_ids, argument_attention_mask, argument_token_type_ids = self._tokenize(
            text=argument, max_len=self.args.statement_max_len
        )
        key_point_input_ids, key_point_attention_mask, key_point_token_type_ids = self._tokenize(
            text=key_point, max_len=self.args.statement_max_len
        )

        stance = torch.tensor([self.stance[idx]], dtype=torch.float)

        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "argument_input_ids": argument_input_ids,
            "argument_attention_mask": argument_attention_mask,
            "argument_token_type_ids": argument_token_type_ids,
            "key_point_input_ids": key_point_input_ids,
            "key_point_attention_mask": key_point_attention_mask,
            "key_point_token_type_ids": key_point_token_type_ids,
            "stance": stance,
        }

        return sample
