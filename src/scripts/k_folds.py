import pandas as pd

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
arg_kp_df = pd.read_csv("IBM_Debater/IBM_Debater_(R)_ArgKP_ACL_2020.v1/ArgKP_dataset.csv")

arg2id = train_df.set_index("topic")["topic_id"].to_dict()
arg2id.update(val_df.set_index("topic")["topic_id"].to_dict())
assert all([arg in arg2id for arg in arg_kp_df["topic"].unique()]), "Some arguments id not found"

arg_kp_df["topic_id"] = arg_kp_df["topic"].map(arg2id)

print(arg_kp_df.tail())
print(train_df.tail())
print(val_df.tail())

print("All topic:")
print(arg_kp_df.topic_id.value_counts(dropna=False))

print("Organizer's train topics")
print(train_df.topic_id.value_counts(dropna=False))

print("Organizer's val topics")
print(val_df.topic_id.value_counts(dropna=False))
