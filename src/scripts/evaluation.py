import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def get_ap(df, label_column, top_percentile=0.5):
    top = int(len(df) * top_percentile)
    df = df.sort_values("score", ascending=False).head(top)
    return average_precision_score(y_true=df[label_column], y_score=df["score"])


def calc_mean_average_precision(df, label_column):
    precisions = [get_ap(group, label_column) for _, group in df.groupby(["topic", "stance"])]
    return np.mean(precisions)


def evaluate_predictions(merged_df):
    mAP_strict = calc_mean_average_precision(merged_df, "label_strict")
    mAP_relaxed = calc_mean_average_precision(merged_df, "label_relaxed")
    print(f"mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")


def load_kpm_data(gold_data_dir, subset):
    arguments_file = os.path.join(gold_data_dir, f"arguments_{subset}.csv")
    key_points_file = os.path.join(gold_data_dir, f"key_points_{subset}.csv")
    labels_file = os.path.join(gold_data_dir, f"labels_{subset}.csv")

    arguments_df = pd.read_csv(arguments_file)
    key_points_df = pd.read_csv(key_points_file)
    labels_file_df = pd.read_csv(labels_file)

    return arguments_df, key_points_df, labels_file_df


def get_predictions(predictions_file, labels_df, arg_df):
    arg_df = arg_df[["arg_id", "topic", "stance"]]
    predictions_df = load_predictions(predictions_file)
    # make sure each arg_id has a prediction
    predictions_df = pd.merge(arg_df, predictions_df, how="left", on="arg_id")

    # handle arguements with no matching key point
    predictions_df["key_point_id"] = predictions_df["key_point_id"].fillna("dummy_id")
    predictions_df["score"] = predictions_df["score"].fillna(0)

    # merge each argument with the gold labels
    merged_df = pd.merge(predictions_df, labels_df, how="left", on=["arg_id", "key_point_id"])

    merged_df.loc[merged_df["key_point_id"] == "dummy_id", "label"] = 0
    merged_df["label_strict"] = merged_df["label"].fillna(0)
    merged_df["label_relaxed"] = merged_df["label"].fillna(1)
    return merged_df


def load_predictions(predictions_dir):
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
        return pd.DataFrame({"arg_id": arg, "key_point_id": kp, "score": scores})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Quantitative Summarization – Key Point Analysis Shared Task official evaluation script"
    )

    parser.add_argument(
        "--gold_data_dir",
        type=str,
        default="kpm_data",
        help="path to the ground truth data directory",
    )

    parser.add_argument(
        "--predictions_file",
        type=str,
        default="outputs/result.json",
        help="path to the json prediction file",
    )

    args = parser.parse_args()

    arg_df, kp_df, labels_df = load_kpm_data(args.gold_data_dir, subset="dev")

    merged_df = get_predictions(args.predictions_file, labels_df, arg_df)
    evaluate_predictions(merged_df)
