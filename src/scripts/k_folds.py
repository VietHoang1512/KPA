import os

import pandas as pd

from src.utils import constants
from src.utils.logging import custom_logger

logger = custom_logger(__name__)


def split_and_dump(all_df: pd.DataFrame, fold_ids: set, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    merged_train = all_df[all_df["topic_id"].map(lambda topic_id: topic_id not in fold_ids)]
    merged_dev = all_df[all_df["topic_id"].map(lambda topic_id: topic_id in fold_ids)]

    arguments_train = merged_train[["arg_id", "argument", "topic", "stance"]].drop_duplicates()
    arguments_dev = merged_dev[["arg_id", "argument", "topic", "stance"]].drop_duplicates()

    key_points_train = merged_train[["key_point_id", "key_point", "topic", "stance"]].drop_duplicates()
    key_points_dev = merged_dev[["key_point_id", "key_point", "topic", "stance"]].drop_duplicates()

    labels_train = merged_train[["arg_id", "key_point_id", "label"]].drop_duplicates()
    labels_dev = merged_dev[["arg_id", "key_point_id", "label"]].drop_duplicates()

    arguments_train.to_csv(os.path.join(output_dir, "arguments_train.csv"), index=False)
    arguments_dev.to_csv(os.path.join(output_dir, "arguments_dev.csv"), index=False)
    key_points_train.to_csv(os.path.join(output_dir, "key_points_train.csv"), index=False)
    key_points_dev.to_csv(os.path.join(output_dir, "key_points_dev.csv"), index=False)
    labels_train.to_csv(os.path.join(output_dir, "labels_train.csv"), index=False)
    labels_dev.to_csv(os.path.join(output_dir, "labels_dev.csv"), index=False)

    logger.info(f"Dump data to {output_dir}")
    logger.info(f"Train size: {len(merged_train)}/ Dev size: {len(merged_dev)}")


if __name__ == "__main__":
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    arg_kp_df = pd.read_csv("IBM_Debater/IBM_Debater_(R)_ArgKP_ACL_2020.v1/ArgKP_dataset.csv")

    arg2id = train_df.set_index("topic")["topic_id"].to_dict()
    arg2id.update(val_df.set_index("topic")["topic_id"].to_dict())
    assert all([arg in arg2id for arg in arg_kp_df["topic"].unique()]), "Some arguments id not found"

    arg_kp_df["topic_id"] = arg_kp_df["topic"].map(arg2id)
    all_df = pd.concat([train_df, val_df])
    logger.warning(
        f"Checking missing data: {all_df[all_df.columns].merge(arg_kp_df, indicator=True, how='outer')['_merge'].eq('both').all()}"
    )

    split_and_dump(all_df=all_df, fold_ids=constants.FOLD1, output_dir="kpm_k_folds/fold_1")
    split_and_dump(all_df=all_df, fold_ids=constants.FOLD2, output_dir="kpm_k_folds/fold_2")
    split_and_dump(all_df=all_df, fold_ids=constants.FOLD3, output_dir="kpm_k_folds/fold_3")
    split_and_dump(all_df=all_df, fold_ids=constants.FOLD4, output_dir="kpm_k_folds/fold_4")
