from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.baselines.data_argument import DataArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


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

    def __len__(self):
        """Denotes the number of examples per epoch."""
        return len(self.data)

    def __getitem__(self, idx):
        """Generate one batch of data."""
        datum: Dict = self.data[idx]

        stance = torch.tensor([datum["stance"]], dtype=torch.float)
        topic = datum["topic"]

        statements = [datum["argument"]] + datum[1] + datum[0]
        label = [0] * (len(datum[1])) + list(range(len(datum[0]) + 1))

        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(text=topic, max_len=self.max_len)
        statements_ecoded = []

        for statement in statements:
            statements_ecoded.append(
                torch.stack(self._tokenize(text=statement, max_len=self.statement_max_len), axis=0)
            )

        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "statements_ecoded": torch.stack(statements_ecoded, axis=0),
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
        statements_ecoded = []

        for statement in statements:
            statements_ecoded.append(
                torch.stack(self._tokenize(text=statement, max_len=self.statement_max_len), axis=0)
            )

        sample = {
            "topic_input_ids": topic_input_ids,
            "topic_attention_mask": topic_attention_mask,
            "topic_token_type_ids": topic_token_type_ids,
            "statements_ecoded": torch.stack(statements_ecoded, axis=0),
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
