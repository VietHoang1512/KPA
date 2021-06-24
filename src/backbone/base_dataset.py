from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.backbone.base_arguments import BaseDataArguments
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


class BaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: BaseDataArguments):
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

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

        input_ids = torch.squeeze(input_ids)
        attention_mask = torch.squeeze(attention_mask)
        token_type_ids = torch.squeeze(token_type_ids)

        if len(inputs.get("overflowing_tokens", [])) > 0:
            logger.warning(f"String `{text}` is truncated with maximum length {max_len}")
        return input_ids, attention_mask, token_type_ids

    def _process_data(self, df: pd.DataFrame) -> List[Dict]:
        raise NotImplementedError
