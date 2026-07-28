import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
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


def plot_epoch(ax, x, y, **kwargs):
    # steps = np.arange(np.min(x), np.max(x)+2)-0.5
    # y = np.insert(y, -1, y[-1])
    # ax.step(steps, y, where="post", **kwargs)
    ax.plot(x, y, **kwargs)

def plot(path, loss, ylabel, label, height=5):

    metrics = read_metrics(path)
    step = metrics["step"]
    epoch = metrics["epoch"]

    early_stopping = get_early_stop(path)

    steps_per_epoch = step[epoch==1][0]
    functions = (lambda epoch: (epoch+0.5)*steps_per_epoch, 
                 lambda step: step/steps_per_epoch - 0.5)

    if not isinstance(loss, list): loss = [loss]
    if not isinstance(ylabel, list): ylabel = [ylabel]
    if not isinstance(height, list): height = [height]
    for i, l in enumerate(loss):
        fig, ax = plt.subplots(figsize=(8/2.54, height[i]/2.54))
        ax_step = ax.secondary_xaxis("top", functions=functions)
        ax.plot(functions[1](step), metrics[f"train_{l}_step"], c="tab:gray")
        mask = np.isfinite(metrics[f"train_{l}_epoch"])
        plot_epoch(ax, epoch[mask], metrics[f"train_{l}_epoch"][mask],
                    c="k", label="Train")
        mask = np.isfinite(metrics[f"val_{l}"])
        plot_epoch(ax, epoch[mask], metrics[f"val_{l}"][mask], 
                   c="tab:red", label="Validation", ls="--")
        if i==0: ax.scatter(*early_stopping, marker="x", c="k", label="Checkpoint", zorder=10)
        ax.legend(frameon=False)
        ax_step.ticklabel_format(axis="x", scilimits=(-3, 3), useMathText=True)
        dir = Path("/home/s/swein/plots/learning_curves")
        save_as = Path(f"{label}_{l}.png") if label is not None else Path("{l}.png")
        fig.savefig(dir / save_as, dpi=300, bbox_inches="tight")
        offset = ax_step.xaxis.offsetText.get_text()
        ax_step.xaxis.offsetText.set_visible(False)
        ax_step.set_xlabel("Iteration ("+offset+")")
        ax.set_xlim(np.min(epoch)-0.5, np.max(epoch)+0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel[i])
        if i!=0: ax.set_yscale("log")
        dir = Path("/home/s/swein/plots/learning_curves")
        save_as = Path(f"{label}_{l}.png") if label is not None else Path("{l}.png")
        fig.savefig(dir / save_as, dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(9/2.54, 6/2.54))
    ax_step = ax.secondary_xaxis("top", functions=functions)
    step_unique, idxs = np.unique(step, return_index=True)
    lr = metrics.get("lr_step_0")[idxs] if metrics.get("lr_step_0") is not None else metrics.get("lr_step")[idxs]
    ax.plot(functions[1](step_unique), lr)
    ax.set_xlim(np.min(epoch), np.max(epoch))
    ax_step.ticklabel_format(axis="x", scilimits=(-3, 3),useMathText=True)
    ax.ticklabel_format(axis="y", scilimits=(-3, 3), useMathText=True)
    dir = Path("/home/s/swein/plots/learning_curves")
    save_as = Path(f"{label}_learning_rate.png") if label is not None else Path("learning_rate.png")
    fig.savefig(dir / save_as, dpi=300, bbox_inches="tight")
    offset = ax_step.xaxis.offsetText.get_text()
    ax_step.xaxis.offsetText.set_visible(False)
    ax_step.set_xlabel("Iteration ("+offset+")")
    ax.set_xlabel("Epoch")
    offset = ax.yaxis.offsetText.get_text()
    ax.yaxis.offsetText.set_visible(False)
    ax.set_ylabel("Learning rate ("+offset+")")
    dir = Path("/home/s/swein/plots/learning_curves")
    save_as = Path(f"{label}_learning_rate.png") if label is not None else Path("learning_rate.png")
    fig.savefig(dir / save_as, dpi=300, bbox_inches="tight")


def read_metrics(path):
    if not isinstance(path, list): path = [path]
    for i, p in enumerate(path):
        print(p)
        p = p / Path("metrics.csv")
        keys = np.loadtxt(
            p, dtype=str, delimiter=",", unpack=True, max_rows=1
        )
        conv = lambda x: float(x) if x else None
        values = np.loadtxt(
            p, delimiter=",", converters=conv, skiprows=1, unpack=True
        )
        if i == 0: 
            metrics = {k: v for k, v in zip(keys, values)}
        else: 
            for k, v in zip(keys, values):
                metrics[k] = np.concatenate((metrics[k], v), axis=-1)
    return metrics


def get_early_stop(path):
    metrics = read_metrics(path)    
    if isinstance(path, list): path = path[-1]
    checkpoints = list((path / Path("checkpoints")).glob("*.ckpt"))
    epochs = [int(ckpt.stem.split("-")[0].split("=")[1]) for ckpt in checkpoints] 
    epoch = min(epochs)
    mask = metrics["epoch"] == epoch
    val_loss = metrics["val_loss"][mask]
    val_loss = val_loss[np.isfinite(val_loss)][0]
    return epoch, val_loss


if __name__ == "__main__":

        path = Path("/home/s/swein/heatmap_regression/logs/version_7")
        plot(path, loss="loss", ylabel="Loss", label="encoder")

        path = Path("/home/s/swein/heatmap_regression/logs/no_encoder/version_0")
        plot(path, loss="loss", ylabel="Loss", label="cnn")

        path = [Path(f"/home/s/swein/detr/logs/version_{i}") for i in [13, 14, 15]]
        plot(path, loss=["loss", "bce_loss", "l1_loss"], height=[5, 4, 4], ylabel=["Loss", "BCE loss", "$L_1$ loss"], label="detr")