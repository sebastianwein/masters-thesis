import h5py
import matplotlib
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


def inset_plot(x, y, z, num_bins = 7):
    z_bins = np.linspace(np.min(z), np.max(z), num_bins+1)
    digitized = np.digitize(z, z_bins)
    digitized = np.clip(digitized, 1, num_bins) - 1
    y_bins = [np.mean(y[digitized==i], axis=0) for i in range(num_bins)]
    num_lines = num_bins 

    colors = matplotlib.colormaps["OrRd"](np.linspace(0.2, 1, num_lines))
    ls = ["--" if i==num_lines-1 else "-" for i in range(num_lines)]
    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))

    handles = list()
    for i in range(num_lines):
        handle, = ax.plot(x, y_bins[i], c=colors[i], ls=ls[i])
        handles.append(handle)

    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(0, 0.15) 

    sample_length = x[-1] - x[0]
    num_tokens = len(x)
    x0, x1 = -6.5/num_tokens*sample_length, 2.5/num_tokens*sample_length
    fig_x0, fig_x1 = 0.525, 0.875
    fig_y0, fig_y1 = 0.125, 0.95
    zoom = ax.inset_axes([fig_x0, fig_y0, fig_x1-fig_x0, fig_y1-fig_y0], xlim=(x0, x1))
    for i in range(num_lines):
        zoom.plot(x, y_bins[i], c=colors[i], ls=ls[i])
    zoom.set_ylim(0)
    inset = ax.indicate_inset_zoom(zoom, edgecolor="black", ls="--", alpha=0.5, lw=0.75)

    x0, x1 = -50, -40
    fig_x0, fig_x1 = 0.075, 0.425
    fig_y0, fig_y1 = 0.125, 0.425
    noise = ax.inset_axes([fig_x0, fig_y0, fig_x1-fig_x0, fig_y1-fig_y0], xlim=(x0, x1))
    mask = np.logical_and(x0<x, x<=x1)
    for i in range(num_lines):
        noise.plot(x[mask], y[i][mask], c=colors[i], ls=ls[i])
    noise.ticklabel_format(axis="y", scilimits=(-1,1))
    # noise.set_yscale("log")
    inset = ax.indicate_inset_zoom(noise, edgecolor="black", ls="--", alpha=0.5, lw=0.75)
    inset.connectors[1].set_visible(False)
    inset.connectors[0].set_visible(True)

    labels = []  # ["No injection"]
    for snr0, snr1 in zip(z_bins, z_bins[1:]):
        space = "  " if snr0<10 else ""
        labels.append(space+rf"{snr0:.1f}$<$SNR$\leq${snr1:.1f}")
    ax.legend(handles, labels, frameon=False, loc="upper left")

    ax.set_xlabel("Time to coalesce")
    ax.set_ylabel("Attention weights")
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/attention/attention.png", dpi=300, bbox_inches="tight")


def double_plot(x, y, z, num_bins = 7):
    z_bins = np.linspace(np.min(z), np.max(z), num_bins+1)
    digitized = np.digitize(z, z_bins)
    digitized = np.clip(digitized, 1, num_bins) - 1
    y_bins = [np.mean(y[digitized==i], axis=0) for i in range(num_bins)]
    num_lines = num_bins 

    colors = matplotlib.colormaps["OrRd"](np.linspace(0.2, 1, num_lines))
    ls = ["--" if i==num_lines-1 else "-" for i in range(num_lines)]
    fig, axs = plt.subplots(ncols=2, width_ratios=(0.6, 0.4), 
                            figsize=(16/2.54, 10/2.54), sharey=True)

    handles = list()
    for i in range(num_lines):
        handle, = axs[0].plot(x, y_bins[i], c=colors[i], ls=ls[i])
        axs[1].plot(x, y_bins[i], c=colors[i], ls=ls[i])
        handles.append(handle)

    axs[0].set_xlim(np.min(x), np.max(x))
    axs[0].set_ylim(0) 
    sample_length = x[-1] - x[0]
    num_tokens = len(x)
    x0, x1 = -6.5/num_tokens*sample_length, 2.5/num_tokens*sample_length
    axs[1].set_xlim(x0, x1)
    axs[1].set_ylim(0) 

    x0, x1 = -50, -40
    fig_x0, fig_x1 = 0.075, 0.425
    fig_y0, fig_y1 = 0.125, 0.425
    noise = axs[0].inset_axes([fig_x0, fig_y0, fig_x1-fig_x0, fig_y1-fig_y0], xlim=(x0, x1))
    mask = np.logical_and(x0<x, x<=x1)
    for i in range(num_lines):
        noise.plot(x[mask], y[i][mask], c=colors[i], ls=ls[i])
    noise.ticklabel_format(axis="y", scilimits=(-1,1))
    # noise.set_yscale("log")
    inset = axs[0].indicate_inset_zoom(noise, edgecolor="black", ls="--", alpha=0.5, lw=0.75)
    inset.connectors[1].set_visible(False)
    inset.connectors[0].set_visible(True)

    labels = []  # ["No injection"]
    for snr0, snr1 in zip(z_bins, z_bins[1:]):
        space = "  " if snr0<10 else ""
        labels.append(space+rf"{snr0:.1f}$<$SNR$\leq${snr1:.1f}")
    axs[0].legend(handles, labels, frameon=False, loc="upper left")

    axs[0].set_xlabel("Time to coalesce")
    axs[1].set_xlabel("Time to coalesce")
    axs[0].set_ylabel("Attention weights")
    fig.subplots_adjust(wspace=0)
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/attention/attention_double.png", dpi=300, bbox_inches="tight")


def uncertainty_plot(x, y, num_bins = 7):
    
    average = np.mean(y, axis=0)
    uncertainty = np.std(y, axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(16/2.54, 9/2.54))

    ax.fill_between(x, average-uncertainty, average+uncertainty, alpha=0.3)
    ax.plot(x, average)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(0, 0.15) 

    sample_length = x[-1] - x[0]
    num_tokens = len(x)
    x0, x1 = -6.5/num_tokens*sample_length, 2.5/num_tokens*sample_length
    fig_x0, fig_x1 = 0.525, 0.875
    fig_y0, fig_y1 = 0.125, 0.95
    zoom = ax.inset_axes([fig_x0, fig_y0, fig_x1-fig_x0, fig_y1-fig_y0], xlim=(x0, x1))
    zoom.fill_between(x, average-uncertainty, average+uncertainty, alpha=0.3)
    zoom.plot(x, average)
    zoom.set_ylim(0)
    inset = ax.indicate_inset_zoom(zoom, edgecolor="black", ls="--", alpha=0.5, lw=0.75)

    x0, x1 = -50, -40
    fig_x0, fig_x1 = 0.075, 0.425
    fig_y0, fig_y1 = 0.125, 0.425
    noise = ax.inset_axes([fig_x0, fig_y0, fig_x1-fig_x0, fig_y1-fig_y0], xlim=(x0, x1))
    mask = np.logical_and(x0<x, x<=x1)
    noise.fill_between(x[mask], average[mask]-uncertainty[mask], average[mask]+uncertainty[mask], alpha=0.3)
    noise.plot(x[mask], average[mask])
    noise.ticklabel_format(axis="y", scilimits=(-1,1))
    # noise.set_yscale("log")
    inset = ax.indicate_inset_zoom(noise, edgecolor="black", ls="--", alpha=0.5, lw=0.75)
    inset.connectors[1].set_visible(False)
    inset.connectors[0].set_visible(True)

    ax.set_xlabel("Time to coalesce")
    ax.set_ylabel("Attention weights")
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/attention/attention_uncertainty.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    metrics_dir = "/scratch/tmp/swein/ggwd/results/binary_out/6000/vit/"
    metrics_files = list(Path(metrics_dir).glob("*.hdf"))
    metrics_files = sorted(metrics_files, key=lambda x: int(x.stem))

    mha = read_dataset(metrics_files[:1], "encoder_cls_weights")
    attention = np.mean(mha, axis=1)[:,1:]

    data_dir = "/scratch/tmp/swein/ggwd/output/binary/6000"
    data_files = list(Path(data_dir).glob("*.hdf"))
    data_files = sorted(data_files, key=lambda x: int(x.stem))
    
    with h5py.File(data_files[0], "r") as file:
        sample_length = file["timeseries"].attrs["sample_length"]
        merger_time = file["parameters/merger_time"][0][0]
    num_tokens = attention.shape[-1]
    time = np.linspace(0, sample_length, num_tokens) - merger_time*sample_length

    snr = read_dataset(data_files[:1], "parameters/nomf_snr").flatten()

    # inset_plot(time, attention, snr)
    uncertainty_plot(time, attention)