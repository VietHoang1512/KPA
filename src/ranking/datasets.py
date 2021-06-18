import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.ranking.data_argument import DataArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


def _extract_id(key_point_id: str) -> float:
    return float(key_point_id.split("_")[-1]) + 1


class RankingTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: DataArguments,
    ):
        """
        Bert Keypoint Argument Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (DataArguments): Data Argument
        """
        df = df.copy()
        self.data = self._process_data(df)
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.statement_max_len = args.statement_max_len
        self.args = args

    def __len__(self):
        """Denotes the number of examples per epoch."""
        return len(self.data)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        datum: Dict = self.data[idx]

        stance = torch.tensor([datum["stance"]], dtype=torch.float)
        topic = datum["topic"]

        random.shuffle(datum[1])

        temp = list(zip(datum[0], datum["neg_class"]))
        random.shuffle(temp)
        datum[0], datum["neg_class"] = zip(*temp)
        datum[0] = list(datum[0])
        datum["neg_class"] = list(datum["neg_class"])

        n_pos = min(len(datum[1]), self.args.max_pos)
        n_neg = min(len(datum[0]), self.args.max_neg)

        statements = [datum["key_point"]] + datum[1][:n_pos] + datum[0][:n_neg]
        label = [datum["class"]] * (n_pos + 1) + datum["neg_class"][:n_neg]
        # print(label)
        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(text=topic, max_len=self.max_len)

        statements_encoded = []
        for statement in statements:
            statements_encoded.append(
                torch.stack(self._tokenize(text=statement, max_len=self.statement_max_len), axis=0)
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

    def _tokenize(self, text: str, max_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_overflowing_tokens=True,
        )
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

        if inputs["num_truncated_tokens"] > 0:
            logger.warning(f"String `{text}` is truncated with maximum length {max_len}")

        return input_ids, attention_mask, token_type_ids

    def _process_data(self, df: pd.DataFrame) -> List[Dict]:
        arg2kp = df[df["label"] == 1].set_index("arg_id")["key_point_id"].map(_extract_id).to_dict()
        df["class"] = df["arg_id"].map(arg2kp).fillna(0)
        data = []
        cnt_neg = 0
        cnt_pos = 0
        for key_point_id, key_point_id_df in df.groupby(["key_point_id"]):
            key_point_id_dict = {0: [], 1: [], "neg_class": []}
            key_point_id_dict.update(key_point_id_df.iloc[0].to_dict())
            key_point_id_dict["class"] = _extract_id(key_point_id)
            for _, row in key_point_id_df.iterrows():
                key_point_id_dict[row["label"]].append(row["argument"])
                if not row["label"]:
                    key_point_id_dict["neg_class"].append(row["class"])
            if len(key_point_id_dict[0]) == 0:
                cnt_neg += 1
            if len(key_point_id_dict[1]) == 0:
                cnt_pos += 1
            data.append(key_point_id_dict)
        logger.warning(
            f"There are {cnt_neg} key points without negative and {cnt_pos} postitive key arguments in total {len(data)}"
        )
        return data


class RankingInferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        arg_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: DataArguments,
    ):
        """
        Bert Keypoint Argument Dataset.

        Args:
            df (pd.DataFrame): Argument-keypoint pairs data frame
            arg_df (pd.DataFrame): DataFrame for all arguments (Used for inference)
            labels_df (pd.DataFrame): DataFrame for labels (Used for inference)
            tokenizer (PreTrainedTokenizer): Pretrained Bert Tokenizer
            args (DataArguments): Data Argument
        """
        df = df.copy()
        self.df = df
        self.arg_df = arg_df.copy()
        self.labels_df = labels_df.copy()
        self.topic = df["topic"].tolist()
        self.argument = df["argument"].tolist()
        self.key_point = df["key_point"].tolist()
        self.label = df["label"].values
        self.stance = df["stance"].values
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.statement_max_len = args.statement_max_len

    def __len__(self):
        """Denotes the number of examples per epoch."""
        return len(self.df)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        topic = self.topic[idx]
        argument = self.argument[idx]
        key_point = self.key_point[idx]

        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(text=topic, max_len=self.max_len)
        statements = [argument, key_point]

        stance = torch.tensor([self.stance[idx]], dtype=torch.float)
        label = [0, 0]

        statements_encoded = []
        for statement in statements:
            statements_encoded.append(
                torch.stack(self._tokenize(text=statement, max_len=self.statement_max_len), axis=0)
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

    def _tokenize(self, text: str, max_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_overflowing_tokens=True,
        )
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

        if inputs["num_truncated_tokens"] > 0:
            logger.warning(f"String `{text}` is truncated with maximum length {max_len}")

        return input_ids, attention_mask, token_type_ids


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
    data_args = DataArguments
    df = pd.read_csv("train.csv")
    dataset = RankingTrainDataset(df, tokenizer, data_args)
    print(dataset[2])
