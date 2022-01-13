import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import AutoConfig, TFAutoModel

from qs_kpa.baselines.model_argument import ModelArguments


def contrastive_loss(y, preds, margin=1):

    y = tf.cast(y, preds.dtype)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)

    return loss


def seed_all(seed=1512):
    """
    Set seed for reproducing result

    Args:
        seed (int, optional): seed number. Defaults to 1512.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)


def build_siamese_model(
    max_len: int,
    arguments_max_len: int,
    args: ModelArguments,
):
    config = AutoConfig.from_pretrained(
        args.model_name,
        from_tf=True,
        output_hidden_states=True,
        output_attentions=True,
    )
    # Bert pretrained model
    bert_model = TFAutoModel.from_pretrained(args.model_name, config=config)

    topic_input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="topic_input_ids")
    topic_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="topic_attention_mask")
    topic_token_type_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="topic_token_type_ids")

    key_point_input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="key_point_input_ids")
    key_point_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="key_point_attention_mask")
    key_point_token_type_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="key_point_token_type_ids")

    argument_input_ids = tf.keras.layers.Input(shape=(arguments_max_len,), dtype=tf.int32, name="argument_input_ids")
    argument_attention_mask = tf.keras.layers.Input(
        shape=(arguments_max_len,), dtype=tf.int32, name="argument_attention_mask"
    )
    argument_token_type_ids = tf.keras.layers.Input(
        shape=(arguments_max_len,), dtype=tf.int32, name="argument_token_type_ids"
    )

    stance = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="stance")

    bert_topic = bert_model(
        topic_input_ids,
        attention_mask=topic_attention_mask,
        token_type_ids=topic_token_type_ids,
    )

    bert_key_point = bert_model(
        key_point_input_ids,
        attention_mask=key_point_attention_mask,
        token_type_ids=key_point_token_type_ids,
    )

    bert_argument = bert_model(
        argument_input_ids,
        attention_mask=argument_attention_mask,
        token_type_ids=argument_token_type_ids,
    )

    topic_output = tf.reduce_mean(tf.stack([bert_topic[2][-i][:, 0, :] for i in range(args.n_hiddens)], axis=1), axis=1)
    key_point_output = tf.reduce_mean(
        tf.stack([bert_key_point[2][-i][:, 0, :] for i in range(args.n_hiddens)], axis=1), axis=1
    )
    argument_output = tf.reduce_mean(
        tf.stack([bert_argument[2][-i][:, 0, :] for i in range(args.n_hiddens)], axis=1), axis=1
    )

    topic_output = tf.keras.layers.Dropout(args.drop_rate)(topic_output)
    key_point_output = tf.keras.layers.Dropout(args.drop_rate)(key_point_output)
    argument_output = tf.keras.layers.Dropout(args.drop_rate)(argument_output)

    topic_keypoint = tf.keras.layers.Concatenate()([topic_output, key_point_output, stance])
    argument = argument_output

    topic_keypoint_rep = tf.keras.layers.Dense(args.text_dim)(topic_keypoint)
    argument_rep = tf.keras.layers.Dense(args.text_dim)(argument)

    inputs = [
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
    sumSquared = K.sum(K.square(topic_keypoint_rep - argument_rep), axis=1, keepdims=True)
    # return the euclidean distance between the vectors

    outputs = K.sqrt(K.maximum(sumSquared, K.epsilon()))

    output = tf.keras.layers.Dense(1, activation="sigmoid")(outputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)

    return model


def build_classification_model(
    max_len: int,
    arguments_max_len: int,
    args: ModelArguments,
):
    config = AutoConfig.from_pretrained(
        args.model_name,
        from_tf=True,
        output_hidden_states=True,
        output_attentions=True,
    )
    # Bert pretrained model
    bert_model = TFAutoModel.from_pretrained(args.model_name, config=config)

    topic_input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="topic_input_ids")
    topic_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="topic_attention_mask")
    topic_token_type_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="topic_token_type_ids")

    key_point_input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="key_point_input_ids")
    key_point_attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="key_point_attention_mask")
    key_point_token_type_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="key_point_token_type_ids")

    argument_input_ids = tf.keras.layers.Input(shape=(arguments_max_len,), dtype=tf.int32, name="argument_input_ids")
    argument_attention_mask = tf.keras.layers.Input(
        shape=(arguments_max_len,), dtype=tf.int32, name="argument_attention_mask"
    )
    argument_token_type_ids = tf.keras.layers.Input(
        shape=(arguments_max_len,), dtype=tf.int32, name="argument_token_type_ids"
    )

    stance = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="stance")

    bert_topic = bert_model(
        topic_input_ids,
        attention_mask=topic_attention_mask,
        token_type_ids=topic_token_type_ids,
    )

    bert_key_point = bert_model(
        key_point_input_ids,
        attention_mask=key_point_attention_mask,
        token_type_ids=key_point_token_type_ids,
    )

    bert_argument = bert_model(
        argument_input_ids,
        attention_mask=argument_attention_mask,
        token_type_ids=argument_token_type_ids,
    )

    topic_output = tf.reduce_mean(tf.stack([bert_topic[2][-i][:, 0, :] for i in range(args.n_hiddens)], axis=1), axis=1)
    key_point_output = tf.reduce_mean(
        tf.stack([bert_key_point[2][-i][:, 0, :] for i in range(args.n_hiddens)], axis=1), axis=1
    )
    argument_output = tf.reduce_mean(
        tf.stack([bert_argument[2][-i][:, 0, :] for i in range(args.n_hiddens)], axis=1), axis=1
    )
    stance_output = tf.keras.layers.Dense(args.stance_dim, activation="relu")(stance)
    topic_output = tf.keras.layers.Dropout(args.drop_rate)(topic_output)
    key_point_output = tf.keras.layers.Dropout(args.drop_rate)(key_point_output)
    argument_output = tf.keras.layers.Dropout(args.drop_rate)(argument_output)

    outputs = tf.keras.layers.Concatenate()([topic_output, key_point_output, argument_output, stance_output])
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(outputs)

    inputs = [
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
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
