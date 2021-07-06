import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score

from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


def get_ap(df: pd.DataFrame, label_column: str, top_percentile: float = 0.5):
    top = int(len(df) * top_percentile)
    df = df.sort_values("score", ascending=False).head(top)
    # after selecting top percentile candidates, we set the score for the dummy kp to 1, to prevent it from increasing the precision.
    df.loc[df["key_point_id"] == "dummy_id", "score"] = 0.99
    return average_precision_score(y_true=df[label_column], y_score=df["score"])


def calc_mean_average_precision(df: pd.DataFrame, label_column: str) -> float:
    precisions = [get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])]
    return np.mean(precisions)


def evaluate_predictions(merged_df: pd.DataFrame):
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
    # logger.info(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")
    return mAP_strict, mAP_relaxed


def load_kpm_data(gold_data_dir: str, subset: str):
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")

    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df


def get_predictions(predictions_file: str, labels_df: pd.DataFrame, arg_df: pd.DataFrame) -> pd.DataFrame:
    arg_df = arg_df[["arg_id", "topic", "stance"]].copy()
    predictions_df = load_predictions(predictions_file)
    # make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

    # handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    # merge each argument with the gold labels
    merged_df = pd.merge(predictions_df, labels_df.copy(), how="left", on=["arg_id", "key_point_id"])

    merged_df.loc[merged_df["key_point_id"] == "dummy_id", "label"] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)
    return merged_df


def load_predictions(predictions_dir: str) -> pd.DataFrame:
    arg = []
    kp = []
    scores = []
    with open(predictions_dir, "r") as f_in:
        res = json.load(f_in)
        for arg_id, kps in res.items():
            best_kp = max(kps.items(), key=lambda x: x[1])
            arg.append(arg_id)
            kp.append(best_kp[0])
            scores.append(best_kp[1])
    print(f"loaded predictions for {len(arg)} arguments")
    return pd.DataFrame({"arg_id": arg, "key_point_id": kp, "score": scores})


def extract_topic(text: str) -> int:
    topic_id = text.split("_")[1]
    return int(topic_id)


def get_data(gold_data_dir: str, subset: str) -> pd.DataFrame:

    logger.info(f"Getting {subset} from {gold_data_dir}")
    arg_df, kp_df, labels_df = load_kpm_data(gold_data_dir, subset)
    arg_df["topic_id"] = arg_df["arg_id"].map(extract_topic)
    kp_df["topic_id"] = kp_df["key_point_id"].map(extract_topic)
    logger.info(f"Num arguments: {len(arg_df)}")
    logger.info(f"Num key points: {len(kp_df)}")
    logger.info(f"Num labelled argument-key point pairs: {len(labels_df)}")

    label_arg_df = pd.merge(labels_df, arg_df, how="left", on="arg_id")
    merged_df = pd.merge(
        label_arg_df,
        kp_df,
        how="left",
        left_on=["key_point_id", "topic", "stance", "topic_id"],
        right_on=["key_point_id", "topic", "stance", "topic_id"],
    )
    assert len(merged_df) == len(labels_df), "Merging dataframes fail"
    return merged_df, arg_df, kp_df, labels_df


def prepare_inference_data(arg_df: pd.DataFrame, kp_df: pd.DataFrame, stance_free: bool = False) -> pd.DataFrame:
    """
    Pair up argmument and keypont to generate score.

    Args:
        arg_df (pd.DataFrame): Arguments dataframe by topic and stance
        kp_df (pd.DataFrame): Keypoints dataframe by topic and stance

    Returns:
        pd.DataFrame: All possible argmument-keypont pair
    """
    topic_id2topic = pd.Series(arg_df.topic, index=arg_df.topic_id).to_dict()
    topic_id2argument = {topic_id: [] for topic_id in arg_df.topic_id.unique()}
    topic_id2keypoint = {topic_id: [] for topic_id in arg_df.topic_id.unique()}
    for _, row in arg_df.iterrows():
        topic_id2argument[row["topic_id"]].append(
            {"argument": row["argument"], "arg_id": row["arg_id"], "stance": row["stance"]}
        )
    for _, row in kp_df.iterrows():
        topic_id2keypoint[row["topic_id"]].append(
            {"key_point": row["key_point"], "key_point_id": row["key_point_id"], "stance": row["stance"]}
        )
    rows = []
    for topic_id, arguments in topic_id2argument.items():
        topic = topic_id2topic[topic_id]
        keypoints = topic_id2keypoint[topic_id]
        for argument in arguments:
            argument_stance = argument["stance"]
            arg_id = argument["arg_id"]
            argument_text = argument["argument"]
            for keypoint in keypoints:
                keypoint_stance = keypoint["stance"]
                if (keypoint_stance != argument_stance) and (not stance_free):
                    continue
                keypoint_text = keypoint["key_point"]
                keypoint_id = keypoint["key_point_id"]
                rows.append([arg_id, keypoint_id, argument_text, keypoint_text, topic, topic_id, argument_stance])
    df = pd.DataFrame(rows, columns=["arg_id", "keypoint_id", "argument", "key_point", "topic", "topic_id", "stance"])
    df["label"] = 0
    return df


def length_plot(lengths: List[int], image_path: Optional[str] = "tmp.png") -> None:
    """
    Plot the sequence length statistic
    Args:
        lengths (List): Sequence lengths (by word or character)
    Returns:
        None
    """
    plt.figure(figsize=(15, 9))
    textstr = f" Mean: {np.mean(lengths):.2f} \u00B1 {np.std(lengths):.2f}  Max: {np.max(lengths)}  Median: {np.median(lengths)}"
    logger.info(image_path + textstr)
    plt.annotate(textstr, xy=(0.1, 0.9), fontsize=14, xycoords="axes fraction")
    sns.countplot(x=lengths, orient="h")
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()
