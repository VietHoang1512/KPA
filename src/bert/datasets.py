import torch
from torch.utils.data import Dataset


class BertKPADataset(Dataset):
    def __init__(self, df, arg_df, labels_df, tokenizer, max_len, argument_max_len, mode):
        self.df = df.copy()
        self.arg_df = arg_df.copy()
        self.labels_df = labels_df.copy()
        self.topic = df["topic"].values
        self.argument = df["argument"].values
        self.key_point = df["key_point"].values
        self.label = df["label"].values
        self.stance = df["stance"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.argument_max_len = argument_max_len
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        topic = self.topic[idx]
        argument = self.argument[idx]
        key_point = self.key_point[idx]

        topic_input_ids, topic_attention_mask, topic_token_type_ids = self._tokenize(text=topic, max_len=self.max_len)
        key_point_input_ids, key_point_attention_mask, key_point_token_type_ids = self._tokenize(
            text=key_point, max_len=self.max_len
        )
        argument_input_ids, argument_attention_mask, argument_token_type_ids = self._tokenize(
            text=argument, max_len=self.argument_max_len
        )
        stance = torch.tensor([self.stance[idx]], dtype=torch.float)

        sample = {
            "topic_input_ids": torch.tensor(topic_input_ids, dtype=torch.long),
            "topic_attention_mask": torch.tensor(topic_attention_mask, dtype=torch.long),
            "topic_token_type_ids": torch.tensor(topic_token_type_ids, dtype=torch.long),
            "key_point_input_ids": torch.tensor(key_point_input_ids, dtype=torch.long),
            "key_point_attention_mask": torch.tensor(key_point_attention_mask, dtype=torch.long),
            "key_point_token_type_ids": torch.tensor(key_point_token_type_ids, dtype=torch.long),
            "argument_input_ids": torch.tensor(argument_input_ids, dtype=torch.long),
            "argument_attention_mask": torch.tensor(argument_attention_mask, dtype=torch.long),
            "argument_token_type_ids": torch.tensor(argument_token_type_ids, dtype=torch.long),
            "stance": stance,
            "label": torch.tensor(self.label[idx], dtype=torch.float),
        }

        return sample

    def _tokenize(self, text: str, max_len: int):
        inputs = self.tokenizer.encode_plus(
            text,
            # add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return input_ids, attention_mask, token_type_ids