import json

from tqdm.auto import tqdm

from src.utils.data import load_kpm_data

arg_df, kp_df, labels_df = load_kpm_data("kpm_data", "dev")
sample_submission = {}
for arg_id in tqdm(arg_df["arg_id"].unique(), desc="Generating pseudo label"):
    random_kp = arg_id.replace("arg", "kp").split("_")[:-1]
    random_kp = "_".join(random_kp)
    sample_submission[arg_id] = {random_kp + "_5": 0.9}

with open("kpm_data/sample_submission.json", "w") as f:
    json.dump(sample_submission, f, indent=4)
