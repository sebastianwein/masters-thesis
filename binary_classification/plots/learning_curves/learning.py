import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

paths = {"cnn_0": Path("/home/s/swein/cnn_vs_vit/conv_logs/0/version_0"),
         "cnn_1000": Path("/home/s/swein/cnn_vs_vit/conv_logs/1000/version_0"),
         "cnn_2000": Path("/home/s/swein/cnn_vs_vit/conv_logs/2000/version_0"),
         "cnn_3000": Path("/home/s/swein/cnn_vs_vit/conv_logs/3000/version_0"),
         "cnn_4000": Path("/home/s/swein/cnn_vs_vit/conv_logs/4000/version_0"),
         "cnn_5000": Path("/home/s/swein/cnn_vs_vit/conv_logs/5000/version_0"),
         "vit_0": Path("/home/s/swein/cnn_vs_vit/logs/version_20"),
         "vit_1000": Path("/home/s/swein/cnn_vs_vit/logs/version_21"),
         "vit_2000": Path("/home/s/swein/cnn_vs_vit/logs/version_22"),
         "vit_3000": Path("/home/s/swein/cnn_vs_vit/logs/version_23"),
         "vit_4000": Path("/home/s/swein/cnn_vs_vit/logs/version_24"),
         "vit_5000": Path("/home/s/swein/cnn_vs_vit/logs/version_25")}
# paths = {"cnn_0": Path("/home/s/swein/cnn_vs_vit/logs/version_26"),
#          "cnn_1000": Path("/home/s/swein/cnn_vs_vit/logs/version_27"),
#          "cnn_2000": Path("/home/s/swein/cnn_vs_vit/logs/version_28"),
#          "cnn_3000": Path("/home/s/swein/cnn_vs_vit/logs/version_29"),
#          "cnn_4000": Path("/home/s/swein/cnn_vs_vit/logs/version_30"),
#          "cnn_5000": Path("/home/s/swein/cnn_vs_vit/logs/version_31"),
#          "vit_0": Path("/home/s/swein/cnn_vs_vit/logs/version_20"),
#          "vit_1000": Path("/home/s/swein/cnn_vs_vit/logs/version_21"),
#          "vit_2000": Path("/home/s/swein/cnn_vs_vit/logs/version_22"),
#          "vit_3000": Path("/home/s/swein/cnn_vs_vit/logs/version_23"),
#          "vit_4000": Path("/home/s/swein/cnn_vs_vit/logs/version_24"),
#          "vit_5000": Path("/home/s/swein/cnn_vs_vit/logs/version_25")}

metrics = dict()
early_stopping = dict()
for model, path in paths.items():
    metrics_path = path / Path("metrics.csv")
    keys = np.loadtxt(
        metrics_path, dtype=str, delimiter=",", unpack=True, max_rows=1
    )
    values = np.loadtxt(
        metrics_path, delimiter=",", 
        converters=lambda x: float(x) if x else None, skiprows=1, unpack=True
    )
    metrics[model] = {k: v for k, v in zip(keys, values)}
    checkpoints = list((path / Path("checkpoints")).glob("*.ckpt"))
    epochs = [int(ckpt.stem.split("-")[0].split("=")[1]) for ckpt in checkpoints] 
    epoch = min(epochs)
    mask = metrics[model]["epoch"] == epoch
    val_loss = metrics[model]["val_loss"][mask]
    val_loss = val_loss[np.isfinite(val_loss)][0]
    early_stopping[model] = (epoch, val_loss)

def plot(ax, metrics, key, xlim=None, function=None, **kwargs):
    x = function(metrics["step"]) if function is not None else metrics["epoch"]
    y = metrics[key]
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]
    mask = np.logical_and(xlim[0] <= x, x <= xlim[1]) if xlim is not None else slice(None)
    x = x[mask]
    y = y[mask]
    ax.plot(x, y, **kwargs)

cnn = ["cnn_0", "cnn_1000", "cnn_2000", "cnn_3000", "cnn_4000", "cnn_5000"]
vit = ["vit_0", "vit_1000", "vit_2000", "vit_3000", "vit_4000", "vit_5000"]
fig, axs = plt.subplots(nrows=len(cnn), ncols=2, figsize=(12/2.54, 10/2.54), sharex=True, sharey=True)
xlim = (0, 50)
for i, key in enumerate(cnn):
    ax = axs[i][0]
    c = f"C{i}"
    plot(ax, metrics[key], "train_loss_epoch", xlim=xlim, ls="--", c=c)
    plot(ax, metrics[key], "val_loss", xlim=xlim, c=c)
    ax.scatter(*early_stopping[key], marker="x", c="k", zorder=10)
for i, key in enumerate(vit):
    ax = axs[i][1]
    c = f"C{i}"
    plot(ax, metrics[key], "train_loss_epoch", xlim=xlim, ls="--", c=c)
    plot(ax, metrics[key], "val_loss", xlim=xlim, c=c)
    ax.scatter(*early_stopping[key], marker="x", c="k", zorder=10)
axs[0,0].set_xlim(xlim)
axs[0,0].set_title("CNN")
axs[0,1].set_title("ViT")
fig.supxlabel("Epoch")
fig.supylabel("BCE Loss")
fig.subplots_adjust(hspace=0, wspace=0)


handles = [
    Line2D([0], [0], color="tab:gray", ls="--", label="Train"),
    Line2D([0], [0], color="tab:gray", ls="-", label="Validation"),
    Line2D([0], [0], color="tab:gray", ls="", marker="x", label="Checkpoint"),
]
ls_legend = fig.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0.77, 0.9),
    frameon=False,
)

labels = ["2s", "4s", "8s", "16s", "32s", "64s"]
handles = [Patch(facecolor=f"C{i}", edgecolor="none", label=l) 
           for i, l in enumerate(labels)]
c_legend = fig.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0.77, 0.75),
    frameon=False,
    title="Sample size",
)
c_legend._legend_box.align = "left"

fig.subplots_adjust(right=0.75)

fig.savefig("/home/s/swein/cnn_vs_vit/plots/learning_curves/curves.png", 
            bbox_inches="tight", dpi=300)
