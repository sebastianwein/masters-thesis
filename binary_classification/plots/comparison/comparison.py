import h5py 
import matplotlib 
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

matplotlib.rc("font", size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc("axes", titlesize=MEDIUM_SIZE)     # fontsize of the axes title
matplotlib.rc("axes", labelsize=SMALL_SIZE)    # fontsize of the x and y labels
matplotlib.rc("xtick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc("ytick", labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc("legend", fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc("figure", titlesize=MEDIUM_SIZE)  # fontsize of the figure title


def read_dataset(paths, dataset_name):
    for i, path in enumerate(paths):
        with h5py.File(path, "r") as file:
            dataset = np.array(file[dataset_name], dtype=np.float64)
            if i == 0: data = dataset
            else: data = np.concatenate((data, dataset))
    return data


paths = {"cnn_0": Path("/scratch/tmp/swein/ggwd/results/binary_out/900/cnn"),
         "cnn_1000": Path("/scratch/tmp/swein/ggwd/results/binary_out/1900/cnn"),
         "cnn_2000": Path("/scratch/tmp/swein/ggwd/results/binary_out/2900/cnn"),
         "cnn_3000": Path("/scratch/tmp/swein/ggwd/results/binary_out/3900/cnn"),
         "cnn_4000": Path("/scratch/tmp/swein/ggwd/results/binary_out/4900/cnn"),
         "cnn_5000": Path("/scratch/tmp/swein/ggwd/results/binary_out/5900/cnn"),
         "vit_0": Path("/scratch/tmp/swein/ggwd/results/binary_out/900/vit"),
         "vit_1000": Path("/scratch/tmp/swein/ggwd/results/binary_out/1900/vit"),
         "vit_2000": Path("/scratch/tmp/swein/ggwd/results/binary_out/2900/vit"),
         "vit_3000": Path("/scratch/tmp/swein/ggwd/results/binary_out/3900/vit"),
         "vit_4000": Path("/scratch/tmp/swein/ggwd/results/binary_out/4900/vit"),
         "vit_5000": Path("/scratch/tmp/swein/ggwd/results/binary_out/5900/vit")}


def compute_roc_score(y, y_pred):
    num_points = 100
    tpr, fpr = np.empty(num_points), np.empty(num_points)
    thresholds = np.linspace(0, 1, num_points)
    for idx, t in enumerate(thresholds):
        pred = y_pred > t
        tp = (pred[y] == 1).sum()
        fn = (pred[y] == 0).sum()
        fp = (pred[np.logical_not(y)] == 1).sum()
        tn = (pred[np.logical_not(y)] == 0).sum()
        fpr[idx] = fp / (fp + tn)
        tpr[idx] = tp / (tp + fn)
    auc = np.trapezoid(x=fpr[::-1], y=tpr[::-1])
    distances = np.sqrt(fpr**2 + (1-tpr)**2)
    idx_opt = np.argmin(distances)
    threshold_opt = thresholds[idx_opt]
    fpr_opt, tpr_opt = fpr[idx_opt], tpr[idx_opt]
    return auc, threshold_opt


def main():

    scores = dict()
    for label, path in paths.items():
        files = list(path.glob("*.hdf"))
        y = np.bool(read_dataset(files, "ground_truth"))
        y_pred = read_dataset(files, "predicitions")
        auc, _ = compute_roc_score(y, y_pred)
        scores[label] = auc
    
    cnn = ["cnn_0", "cnn_1000", "cnn_2000", "cnn_3000", "cnn_4000", "cnn_5000"]
    vit = ["vit_0", "vit_1000", "vit_2000", "vit_3000", "vit_4000", "vit_5000"]
    label = ["2", "4", "8", "16", "32", "64"]
    fig, ax = plt.subplots(figsize=(8/2.54, 5/2.54))
    x = np.arange(len(label))
    width = 0.35
    ax.bar(x-width/2, [scores[k] for k in cnn], width, label="CNN")
    ax.bar(x+width/2, [scores[k] for k in vit], width, label="ViT")
    lower, upper = np.min(list(scores.values())), np.max(list(scores.values()))
    delta = upper - lower
    ax.set_ylim(lower-0.1*delta, upper+0.1*delta)
    ax.set_xticks(x, label)
    ax.legend(frameon=False)
    ax.set_xlabel("Sample length (s)")
    ax.set_ylabel("AUROC")
    fig.tight_layout()
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/comparison/performance.png", 
                dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()