import pandas as pd

from qs_kpa import KeyPointAnalysis
from qs_kpa.train_utils.distance import SiameseDistanceMetric
from qs_kpa.utils.data import evaluate_predictions, get_data, prepare_inference_data
from qs_kpa.utils.logging import custom_logger

logger = custom_logger(__name__)


def evaluate(data_dir: str, encoder: KeyPointAnalysis):
    df, arg_df, kp_df, labels_df = get_data(gold_data_dir=data_dir, subset="dev")
    inf_df = prepare_inference_data(arg_df, kp_df)

    arguments = []
    key_points = []

    for _, row in inf_df.iterrows():
        argument = (row["topic"], row["argument"], row["stance"])
        key_point = (row["topic"], row["key_point"], row["stance"])
        arguments.append(argument)
        key_points.append(key_point)
    arguments_emb = encoder.encode(arguments, convert_to_tensor=True)
    key_points_emb = encoder.encode(key_points, convert_to_tensor=True)
    inf_df["label"] = 1 - SiameseDistanceMetric.COSINE_DISTANCE(arguments_emb, key_points_emb).cpu()

    arg_df = arg_df[["arg_id", "topic", "stance"]].copy()

    predictions = {arg_id: dict() for arg_id in df["arg_id"].unique()}
    for _, row in inf_df.iterrows():
        predictions[row["arg_id"]][row["keypoint_id"]] = row["label"]

    arg = []
    kp = []
    scores = []
    for arg_id, kps in predictions.items():
        best_kp = max(kps.items(), key=lambda x: x[1])
        arg.append(arg_id)
        kp.append(best_kp[0])
        scores.append(best_kp[1])

    predictions_df = pd.DataFrame({"arg_id": arg, "key_point_id": kp, "score": scores})
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

    return evaluate_predictions(merged_df)


if __name__ == "__main__":
    encoder = KeyPointAnalysis(from_pretrained=False)
    pretrained_encoder = KeyPointAnalysis(from_pretrained=True)
    mAP_strict, mAP_relaxed = evaluate("kpm_k_folds/fold_2", encoder=encoder)
    logger.info(f"Model using roBERTa directly: mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")
    mAP_strict, mAP_relaxed = evaluate("kpm_k_folds/fold_2", encoder=pretrained_encoder)
    logger.info(f"Our pretrained model: mAP strict= {mAP_strict} ; mAP relaxed = {mAP_relaxed}")
