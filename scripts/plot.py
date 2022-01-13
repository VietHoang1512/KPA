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
        if col in ["model", "Avg mAP strict", "Avg mAP relaxed"]:
            continue
        fold, metric = col.strip().split(" - ")
        fold_id = int(fold[-1])
        result[metric][model][fold_id - 1] = row[col]

for model in MODEL:
    print(model)
    print(f'Avg mAP strict: {np.mean(result["strict"][model])} \u00B1 {np.std(result["strict"][model])}')
    print(f'Avg mAP relaxed: {np.mean(result["relax"][model])} \u00B1 {np.std(result["relax"][model])}')

FOLDS = [f"Fold {i}" for i in range(1, 8)]

fig, ax = plt.subplots(2, 1, figsize=(7, 8))

for i, metric in enumerate(result):
    ax = plt.subplot(2, 1, i + 1)
    ax.set_title((metric + "ed" if metric == "relax" else metric) + " mAP")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.plot(FOLDS, result[metric]["base line w/o miner"], label="SimAKP", marker="*", linestyle="--")
    plt.plot(FOLDS, result[metric]["qa"], label="QA", marker="^", linestyle="--")
    plt.plot(FOLDS, result[metric]["mixed-loss"], label="T-SimAKP", marker="x", linestyle="--")
    plt.plot(FOLDS, result[metric]["pseudo label-triplet"], label="T-MTS", marker="D", linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    plt.grid()

plt.legend(handles, labels, ncol=4, loc="lower left")
fig = plt.gcf()
# fig.suptitle("Mean Average Precision over 7 folds", fontsize=14, x=0.51, y=1.1)
plt.savefig("7-folds.pdf", bbox_inches="tight")
plt.show()

map_strict = [
    np.mean(result["strict"]["pseudo label-triplet"]),
    np.mean(result["strict"]["pseudo label-tup"]),
    np.mean(result["strict"]["pseudo label-tup w/o miner"]),
    np.mean(result["strict"]["pseudo label-tup w. batch_norm"]),
]
map_strict_std = [
    np.std(result["strict"]["pseudo label-triplet"]),
    np.std(result["strict"]["pseudo label-tup"]),
    np.std(result["strict"]["pseudo label-tup w/o miner"]),
    np.std(result["strict"]["pseudo label-tup w. batch_norm"]),
]
map_relaxed = [
    np.mean(result["relax"]["pseudo label-triplet"]),
    np.mean(result["relax"]["pseudo label-tup"]),
    np.mean(result["relax"]["pseudo label-tup w/o miner"]),
    np.mean(result["relax"]["pseudo label-tup w. batch_norm"]),
]
map_relaxed_std = [
    np.std(result["relax"]["pseudo label-triplet"]),
    np.std(result["relax"]["pseudo label-tup"]),
    np.std(result["relax"]["pseudo label-tup w/o miner"]),
    np.std(result["relax"]["pseudo label-tup w. batch_norm"]),
]

ind = ["T-MTS", "MTS", "MTS w/o miner", "MTS w. BN"]
fig, ax = plt.subplots(figsize=(8, 6))
width = 0.3
p1 = ax.bar(ind, map_strict, width, yerr=map_strict_std, color="dodgerblue")
p2 = ax.bar(ind, map_relaxed, width, bottom=map_strict, yerr=map_relaxed_std, color="tomato")
# plt.xlabel("Model", fontsize=20)
# plt.ylabel("Average score", fontsize=20)
# plt.title("Mean Average Precision score on ArgKP for different modifications of MTS")


def autolabel(i, rects, std):
    """
    Attach a text label above each bar displaying its height
    """
    for k, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            (i + 0.45) * height,
            f"{height:.2f}\u00B1{std[k]:.2f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )


ax.set_xticklabels(ind, fontsize=15)
autolabel(0, p1, map_strict_std)
autolabel(1, p2, map_relaxed_std)
plt.legend(
    (p1[0], p2[0]),
    ("strict mAP", "relaxed mAP"),
    loc="upper center",
    prop={"size": 12},
    ncol=2,
    bbox_to_anchor=(0.47, 0.0, 0.1, 1.0),
)
plt.ylim(0, 2.0)
plt.xlim(-0.5, 3.5)
plt.savefig("modification.pdf", bbox_inches="tight")
plt.show()
