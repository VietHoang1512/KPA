import argparse

import pandas as pd

parser = argparse.ArgumentParser(description="Data Preparation for Mask language model")

parser.add_argument(
    "--arg_kp",
    type=str,
    default="IBM_Debater/IBM_Debater_(R)_ArgKP_ACL_2020.v1/ArgKP_dataset.csv",
    help="path to the ArgKP csv data file",
)

parser.add_argument(
    "--argq_30k",
    type=str,
    default="IBM_Debater/arg_quality_rank_30k.csv",
    help="path to the ARGQ Rank 30kArgs csv file",
)

args = parser.parse_args()

if __name__ == "__main__":
    arg_kp_df = pd.read_csv(args.arg_kp)
    argq_30k_df = pd.read_csv(args.argq_30k)
    arg_kp_texts = arg_kp_df["topic"].tolist() + arg_kp_df["key_point"].tolist() + arg_kp_df["argument"].tolist()
    argq_30k_texts = argq_30k_df["topic"].tolist() + argq_30k_df["argument"].tolist()
    texts = set(arg_kp_texts + argq_30k_texts)
    df = pd.DataFrame(texts, columns=["text"])
    df.to_csv("mlm_data.csv")
    print(df.tail())
