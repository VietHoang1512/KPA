import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import PreTrainedTokenizer

from src.bert.data_argument import DataArguments


class BertKPAGenerator(tf.keras.utils.Sequence):
    """Data Generator for Keras model."""

    def __init__(
        self,
        shuffle,
        batch_size,
        df: pd.DataFrame,
        arg_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        args: DataArguments,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.labels = df["label"].values.astype(float)

        self.df = df.copy()
        self.arg_df = arg_df.copy()
        self.labels_df = labels_df.copy()

        topics = df["topic"].tolist()

        self.topic_encoded = tokenizer.batch_encode_plus(
            topics,
            return_token_type_ids=True,
            padding="max_length",
            max_length=args.max_len,
            truncation=True,
        )

        key_points = df["key_point"].tolist()

        self.key_point_encoded = tokenizer.batch_encode_plus(
            key_points,
            return_token_type_ids=True,
            padding="max_length",
            max_length=args.max_len,
            truncation=True,
        )

        arguments = df["argument"].tolist()

        self.argument_encoded = tokenizer.batch_encode_plus(
            arguments,
            return_token_type_ids=True,
            padding="max_length",
            max_length=args.argument_max_len,
            truncation=True,
        )

        self.label = df["label"].astype(float).values
        self.stance = df["stance"].astype(float).values

        self.total = len(df)
        self.indexes = np.arange(self.total)
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(self.total / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data."""
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        labels = np.array([self.labels[k] for k in indexes])

        topic_input_ids = tf.convert_to_tensor([self.topic_encoded["input_ids"][k] for k in indexes])
        topic_attention_mask = tf.convert_to_tensor([self.topic_encoded["attention_mask"][k] for k in indexes])
        topic_token_type_ids = tf.convert_to_tensor([self.topic_encoded["token_type_ids"][k] for k in indexes])

        key_point_input_ids = tf.convert_to_tensor([self.key_point_encoded["input_ids"][k] for k in indexes])
        key_point_attention_mask = tf.convert_to_tensor([self.key_point_encoded["attention_mask"][k] for k in indexes])
        key_point_token_type_ids = tf.convert_to_tensor([self.key_point_encoded["token_type_ids"][k] for k in indexes])

        argument_input_ids = tf.convert_to_tensor([self.argument_encoded["input_ids"][k] for k in indexes])
        argument_attention_mask = tf.convert_to_tensor([self.argument_encoded["attention_mask"][k] for k in indexes])
        argument_token_type_ids = tf.convert_to_tensor([self.argument_encoded["token_type_ids"][k] for k in indexes])

        stance = tf.convert_to_tensor([self.stance[k] for k in indexes])

        features = [
            topic_input_ids,
            topic_attention_mask,
            topic_token_type_ids,
            key_point_input_ids,
            key_point_attention_mask,
            key_point_token_type_ids,
            argument_input_ids,
            argument_attention_mask,
            argument_token_type_ids,
            stance,
        ]

        return features, labels
