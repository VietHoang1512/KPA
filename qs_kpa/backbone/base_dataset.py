from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from qs_kpa.backbone.base_arguments import BaseDataArguments
from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


class BaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: BaseDataArguments):
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer_type = type(tokenizer).__name__.lower().replace("tokenizer", "").replace("fast", "")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _tokenize(self, text: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if self.tokenizer_type in ["xlnet", "albert"]:
            if input_ids.size(0) > 1:
                logger.warning(f"String `{text}` is truncated with maximum length {max_len}")
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
            token_type_ids = token_type_ids[0]
        else:
            if inputs.get("num_truncated_tokens", 0) > 0:
                logger.warning(f"String `{text}` is truncated with maximum length {max_len}")

        return input_ids, attention_mask, token_type_ids

    def _process_data(self, df: pd.DataFrame) -> List[Dict]:
        raise NotImplementedError
