import json
import os

PREDICT_DIR = "predict"

predictions = []
for seed in os.listdir(PREDICT_DIR):
    fp = os.path.join(PREDICT_DIR, seed, "predictions.p")
    with open(fp, "r") as f:
        prediction = json.load(f)
    predictions.append(prediction)
final = predictions[0]
for prediction in predictions[1:]:
    for arg_id, kps in prediction.items():
        for kp_id, score in kps.items():
            final[arg_id][kp_id] += score
for arg_id, kps in final.items():
    for kp_id, score in kps.items():
        final[arg_id][kp_id] = final[arg_id][kp_id] / (4 * len(os.listdir(PREDICT_DIR)))
with open(os.path.join(PREDICT_DIR, "predictions.p"), "w") as f:
    json.dump(final, f, indent=4)
