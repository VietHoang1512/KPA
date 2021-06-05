import json

from src.utils.data import evaluate_predictions, get_predictions, load_kpm_data

arguments_df, key_points_df, labels_file_df = load_kpm_data(gold_data_dir="kpm_data", subset="dev")

submission = {arg_id: {} for arg_id in arguments_df.arg_id.unique()}
for _, row in labels_file_df.iterrows():
    submission[row["arg_id"]][row["key_point_id"]] = row["label"]

with open("kpm_data/predictions.p", "w") as f:
    json.dump(submission, f, indent=4)

merged_df = get_predictions("kpm_data/predictions.p", labels_file_df, arguments_df)
print(evaluate_predictions(merged_df))
