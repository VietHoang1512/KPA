import pandas as pd

from src.utils.data import extract_topic

arguments_test_df = pd.read_csv("kpm_6_folds/test/arguments_test.csv")
key_points_test_df = pd.read_csv("kpm_6_folds/test/key_points_test.csv")

arguments_test_df["topic_id"] = arguments_test_df["arg_id"].map(extract_topic)
key_points_test_df["topic_id"] = key_points_test_df["key_point_id"].map(extract_topic)

label_df = pd.merge(arguments_test_df, key_points_test_df, how="left", on=["topic", "stance", "topic_id"])[
    ["arg_id", "key_point_id"]
]
label_df["label"] = 0

label_df.to_csv("kpm_6_folds/test/labels_test.csv", index=False)
