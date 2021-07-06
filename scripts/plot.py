import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("assets/metric.csv", header=0)

columns = data.columns
MODEL = data["model"].tolist()

result = dict()
result["strict"] = {model: [0] * 7 for model in MODEL}
result["relax"] = {model: [0] * 7 for model in MODEL}

for _, row in data.iterrows():
    model = row["model"]
    for col in columns:
        if col == "model":
            continue
        fold, metric = col.strip().split(" - ")
        fold_id = int(fold[-1])
        result[metric][model][fold_id - 1] = row[col]

FOLDS = [f"Fold {i}" for i in range(1, 8)]

fig, ax = plt.subplots(1, 2, figsize=(16, 4))

for i, metric in enumerate(result):
    ax = plt.subplot(1, 2, i + 1)
    ax.set_title("mAP (" + (metric + "ed" if metric == "relax" else metric) + ")")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.plot(FOLDS, result[metric]["base line"], label="AKP", marker="*", linestyle="--")
    plt.plot(FOLDS, result[metric]["qa"], label="QA", marker="^", linestyle="--")
    plt.plot(FOLDS, result[metric]["mixed-loss"], label="T-AKP", marker="x", linestyle="--")
    plt.plot(FOLDS, result[metric]["pseudo label-tup"], label="MTS", marker="D", linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    plt.grid()

plt.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(-0.11, 1.2))
fig = plt.gcf()
fig.suptitle("Mean Average Precision over 7 folds", fontsize=14, x=0.51, y=1.1)
plt.savefig("7-folds.pdf", bbox_inches="tight")
plt.show()

map_strict = [
    np.mean(result["strict"]["pseudo label-tup"]),
    np.mean(result["strict"]["pseudo label-tup w/o miner"]),
    np.mean(result["strict"]["pseudo label-tup w. batch_norm"]),
    np.mean(result["strict"]["pseudo label-tup w. normalization"]),
]
map_strict_std = [
    np.std(result["strict"]["pseudo label-tup"]),
    np.std(result["strict"]["pseudo label-tup w/o miner"]),
    np.std(result["strict"]["pseudo label-tup w. batch_norm"]),
    np.std(result["strict"]["pseudo label-tup w. normalization"]),
]
map_relaxed = [
    np.mean(result["relax"]["pseudo label-tup"]),
    np.mean(result["relax"]["pseudo label-tup w/o miner"]),
    np.mean(result["relax"]["pseudo label-tup w. batch_norm"]),
    np.mean(result["relax"]["pseudo label-tup w. normalization"]),
]
map_relaxed_std = [
    np.std(result["relax"]["pseudo label-tup"]),
    np.std(result["relax"]["pseudo label-tup w/o miner"]),
    np.std(result["relax"]["pseudo label-tup w. batch_norm"]),
    np.std(result["relax"]["pseudo label-tup w. normalization"]),
]

ind = ["MTS", "MTS w/o miner", "MTS w. BN", "MTS w. norm"]
fig, ax = plt.subplots(figsize=(12, 7))
width = 0.15
p1 = ax.bar(ind, map_strict, width, yerr=map_strict_std, color="royalblue")
p2 = ax.bar(ind, map_relaxed, width, bottom=map_strict, yerr=map_relaxed_std, color="mediumvioletred")
plt.xlabel("Model")
plt.ylabel("Average score")
plt.title("Mean Average Precision score on ArgKP for different modifications of MTS")


def autolabel(i, rects, std):
    """
    Attach a text label above each bar displaying its height
    """
    for k, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            (i + 0.45) * height,
            f"{height:.3f} \u00B1 {std[k]:.3f}",
            ha="center",
            va="bottom",
        )


autolabel(0, p1, map_strict_std)
autolabel(1, p2, map_relaxed_std)
plt.legend((p1[0], p2[0]), ("mAP (strict)", "mAP (relaxed)"), loc="best")
plt.savefig("modification.pdf", bbox_inches="tight")
plt.show()
