from glob import glob
import h5py
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from uncertainties import ufloat, unumpy

from util import *


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


def far_plot(y_list, y_pred_list, number_of_signals, total_sample_length, desired_far, labels, zoom = None, save_as = "far"):

    if zoom is None: zoom = [True for i in range(len(y_list))]

    far, detection_ratio = list(), list()
    threshold_opt, far_opt, detection_ratio_opt = list(), list(), list()
    for y, y_pred in zip(y_list, y_pred_list):
        out = compute_far(y, y_pred, number_of_signals, total_sample_length, desired_far)
        far.append(out[0])
        detection_ratio.append(out[1])
        threshold_opt.append(out[2])
        far_opt.append(out[3])
        detection_ratio_opt.append(out[4])

    fig, ax = plt.subplots(figsize=(9/2.54, 9/2.54))
    zoom_ax = ax.inset_axes((0.6, 0.1, 0.3, 0.3))
    ax.indicate_inset_zoom(zoom_ax, ls="--")
    for i in range(len(y_list)):
        ax.plot(far[i], detection_ratio[i], label=labels[i], c=f"C{i}")
        if not zoom[i]: continue
        zoom_ax.plot(far[i], detection_ratio[i], c=f"C{i}")
        ax.scatter(far_opt[i], detection_ratio_opt[i], c="tab:red", zorder=10)
        zoom_ax.scatter(far_opt[i], detection_ratio_opt[i], c="tab:red", zorder=10)
        zoom_ax.annotate(f"{round(100*threshold_opt[i]):.1f}%", (far_opt[i], detection_ratio_opt[i]), 
                         xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("False alarm rate (1/s)")
    ax.set_ylabel("Detection ratio")
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    ax.set_xlim(3.154e-7, 16/64)

    y0 = min([r for i, r in enumerate(detection_ratio_opt) if zoom[i]])
    y1 = max([r for i, r in enumerate(detection_ratio_opt) if zoom[i]])
    delta = 1.5 *(y1 - y0)
    zoom_ax.set_ylim(y0-delta, y1+delta)
    zoom_ax.set_xscale("log")
    a = 1.1
    zoom_ax.set_xlim(desired_far/a, desired_far*a)
    zoom_ax.plot([desired_far]*2, [y0-delta, y1], ls="--", color="k", lw=0.75)
    zoom_ax.plot([desired_far/a, desired_far], [y0]*2, ls="--", color="k", lw=0.75)
    zoom_ax.plot([desired_far/a, desired_far], [y1]*2, ls="--", color="k", lw=0.75)
    zoom_ax.set_yticks([y0, y1], [f"{(100*y0):.1f}%", f"{(100*y1):.1f}%"])
    zoom_ax.set_xticks([desired_far], [f"1/{round(1/(desired_far*60))}min"])
    zoom_ax.xaxis.set_tick_params(which="minor", bottom=False, labelbottom=False)

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"/home/s/swein/plots/{save_as}.png", dpi=300, bbox_inches="tight")

    return threshold_opt


def bar_plot(x, y, edge=True, annotate=True):
    fig, ax = plt.subplots(figsize=(10/2.54, 7/2.54))
    widths = (x[1:] - x[:-1])
    p = ax.bar(x[:-1], y, widths, align="edge", edgecolor="k" if edge else None)
    if annotate:
        y0, y1 = ax.get_ylim()
        yrange = y1 - y0
        labels = [f"{int(100*height)}" if height/yrange > 0.05 else "" for height in y]
        ax.bar_label(p, labels, label_type="center", color="white")
    ax.set_xlim(np.min(x), np.max(x))
    return fig, ax


def snr_plot(tp, snr, num_bins = 10, save_as = "snr"):
    snr_bins = np.linspace(np.min(snr), np.max(snr), num_bins)
    digitized = np.digitize(snr, snr_bins)
    digitized = np.clip(digitized, 1, num_bins-1) - 1
    detection_ratio = [tp[digitized==i].sum()/(digitized==i).sum() 
                       for i in range(num_bins-1)]
    fig, ax = bar_plot(snr_bins, detection_ratio)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Optimal SNR")
    ax.set_ylabel("Detection ratio")
    fig.tight_layout()
    fig.savefig(f"/home/s/swein/plots/{save_as}.png", dpi=300, bbox_inches="tight")


def time_plot(time, time_pred, num_bins = 20, normalize=True, save_as = "time"):
    delta = 1000*(time - time_pred)
    mean, sd = np.mean(delta), np.std(delta, ddof=1)
    delta_bins = np.linspace(-100, 100, num_bins)
    digitized = np.digitize(delta, delta_bins) - 1
    norm = len(time) if normalize else 1
    total = [(digitized==i).sum()/norm for i in range(num_bins-1)]
    fig, ax = bar_plot(delta_bins, total, edge=False, annotate=False)
    ax.axvline(mean, c="k", ls="--", lw=1)
    y0, y1 = ax.get_ylim()
    y = y0 + (y1-y0)/2
    ax.errorbar(mean, y, xerr=sd, capsize=4, c="k", elinewidth=1, capthick=1)
    handle = mpatches.Rectangle((0, 0), 0, 0, alpha=0)
    handles = [handle, handle]
    labels = [f"$\mu={int(mean)}\,$ms", f"$\sigma={int(sd)}\,$ms"]
    ax.legend(handles, labels, frameon=False)
    ax.set_xlabel(r"Prediction error $\hat{t}-t$ (ms)")
    ylabel = "True positives (%)" if normalize else "True positives"
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(f"/home/s/swein/plots/{save_as}.png", dpi=300, bbox_inches="tight")


def num_mergers_plot(tps, num_mergers, labels, save_as):

    def ratio(tp, num_mergers):
        unique = np.unique(num_mergers[num_mergers>0])
        detection_ratio = [tp[num_mergers==n].sum() / ((num_mergers==n).sum()*n)
                           for n in unique]
        std_dev = [np.sqrt(p*(1-p) / ((num_mergers==n).sum()*n))
                   for p, n in zip(detection_ratio, unique)]
        return unique, detection_ratio, std_dev
    
    ratios = list()
    err = list()
    for tp in tps:
        unique, r, sd = ratio(tp, num_mergers)
        ratios.append(r)
        err.append(sd)
        print([f"{x:.1uS}" for x in unumpy.uarray(r, sd)])
        i, j = np.argmin(r), np.argmax(r)
        x = ufloat(r[j], sd[j]) - ufloat(r[i], sd[i])
        print(f"{x:.1uS}")


    fig, ax = plt.subplots(figsize=(12/2.54, 6/2.54))
    x = np.arange(len(unique))
    width = 0.8
    for i, (r, sd) in enumerate(zip(ratios, err)):
        w = width / len(ratios)
        p = ax.bar(x-width/2+i*w, r, w,
                   align="edge", label=labels[i], yerr=sd, capsize=4, error_kw={"lw": 1})
        # label = [f"{round(100*x)}" for x in cnn_ratio]
        # ax.bar_label(p, label, c="white", label_type="center")

    lower = np.min(ratios) 
    upper = np.max(ratios)
    delta = upper - lower
    ax.set_ylim(lower-0.3*delta, upper+0.1*delta)
    ax.set_xticks(x, unique)
    ax.legend(frameon=False)
    ax.set_xlabel("Number of signals per sample")
    ax.set_ylabel("Detection ratio")
    fig.tight_layout()
    fig.savefig(f"/home/s/swein/plots/{save_as}.png", 
                dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    N = 32*4096

    cnn_dir = "/scratch/tmp/swein/ggwd/results/heatmap_no_encoder_out/300"
    cnn_files = list(Path(cnn_dir).glob("*.hdf"))
    cnn_files = sorted(cnn_files, key=lambda x: int(x.stem))

    encoder_dir = "/scratch/tmp/swein/ggwd/results/heatmap_out/300"
    encoder_files = list(Path(encoder_dir).glob("*.hdf"))
    encoder_files = sorted(encoder_files, key=lambda x: int(x.stem))

    detr_dir = "/scratch/tmp/swein/ggwd/results/detr_out/300"
    detr_files = list(Path(detr_dir).glob("*.hdf"))
    detr_files = sorted(detr_files, key=lambda x: int(x.stem))

    data_dir = "/scratch/tmp/swein/ggwd/output/test/300"
    data_files = list(Path(data_dir).glob("*.hdf"))
    data_files = sorted(data_files, key=lambda x: int(x.stem))

    time = read_dataset(data_files, "parameters/merger_time", max_rows=N)
    snr = read_dataset(data_files, "parameters/nomf_snr", max_rows=N)
    n = time.shape[0]
    number_of_signals = np.isfinite(time).sum()
    sample_length = 64
    total_sample_length = len(time)*sample_length
    desired_far = 1/(10*60)
    sigma = 0.1/sample_length
    
    cnn_y_pred = read_dataset(cnn_files, "y_pred", max_rows=N)
    cnn_t_pred = read_dataset(cnn_files, "t_pred", max_rows=N)
    cnn_y, cnn_signal_idx, cnn_pred_idx = greedy_match(cnn_y_pred, cnn_t_pred, time, sigma)
    cnn_threshold = compute_far(cnn_y, cnn_y_pred, number_of_signals, total_sample_length, desired_far)[2]
    print(cnn_threshold)
    
    tp = (match(cnn_y_pred, cnn_pred_idx) > cnn_threshold).flatten()
    matched_snr = snr.flatten()
    mask = np.isfinite(matched_snr)
    snr_plot(tp[mask], matched_snr[mask], save_as="cnn_snr")
    matched_time = time.flatten()*sample_length
    matched_time_pred = match(cnn_t_pred, cnn_pred_idx).flatten()*sample_length
    time_plot(matched_time_pred[tp], matched_time[tp], save_as="cnn_time")

    encoder_y_pred = read_dataset(encoder_files, "y_pred", max_rows=N)
    encoder_t_pred = read_dataset(encoder_files, "t_pred", max_rows=N)
    encoder_y, encoder_signal_idx, encoder_pred_idx = greedy_match(encoder_y_pred, encoder_t_pred, time, sigma)
    encoder_threshold = compute_far(encoder_y, encoder_y_pred, number_of_signals, total_sample_length, 3.8E-7)[2]
    print(encoder_threshold)
    
    tp = (match(encoder_y_pred, encoder_pred_idx) > encoder_threshold).flatten()
    matched_snr = snr.flatten()
    mask = np.isfinite(matched_snr)
    snr_plot(tp[mask], matched_snr[mask], save_as="encoder_low_far_snr")
    matched_time = time.flatten()*sample_length
    matched_time_pred = match(encoder_t_pred, encoder_pred_idx).flatten()*sample_length
    time_plot(matched_time_pred[tp], matched_time[tp], save_as="encoder_low_far_time")

    encoder_y_pred = read_dataset(encoder_files, "y_pred", max_rows=N)
    encoder_t_pred = read_dataset(encoder_files, "t_pred", max_rows=N)
    encoder_y, encoder_signal_idx, encoder_pred_idx = greedy_match(encoder_y_pred, encoder_t_pred, time, sigma)
    encoder_threshold = compute_far(encoder_y, encoder_y_pred, number_of_signals, total_sample_length, desired_far)[2]
    print(encoder_threshold)
    
    tp = (match(encoder_y_pred, encoder_pred_idx) > encoder_threshold).flatten()
    matched_snr = snr.flatten()
    mask = np.isfinite(matched_snr)
    snr_plot(tp[mask], matched_snr[mask], save_as="encoder_snr")
    matched_time = time.flatten()*sample_length
    matched_time_pred = match(encoder_t_pred, encoder_pred_idx).flatten()*sample_length
    time_plot(matched_time_pred[tp], matched_time[tp], save_as="encoder_time")

    detr_y_pred = read_dataset(detr_files, "cls_pred", max_rows=N)
    detr_t_pred = read_dataset(detr_files, "time_pred", max_rows=N)
    detr_y, detr_signal_idx, detr_pred_idx = greedy_match(detr_y_pred, detr_t_pred, time, sigma)
    detr_threshold = compute_far(detr_y, detr_y_pred, number_of_signals, total_sample_length, desired_far)[2]
    print(detr_threshold)
    
    tp = (match(detr_y_pred, detr_pred_idx) > detr_threshold).flatten()
    matched_snr = snr.flatten()
    mask = np.isfinite(matched_snr)
    snr_plot(tp[mask], matched_snr[mask], save_as="detr_snr")
    matched_time = time.flatten()*sample_length
    matched_time_pred = match(detr_t_pred, detr_pred_idx).flatten()*sample_length
    time_plot(matched_time_pred[tp], matched_time[tp], save_as="detr_time")

    y = (cnn_y, encoder_y)
    y_pred = (cnn_y_pred, encoder_y_pred)
    far_plot(
        y, y_pred, number_of_signals, total_sample_length, desired_far, ("CNN", "Encoder"), save_as="cnn_encoder_far"
    )
    y = (encoder_y, detr_y)
    y_pred = (encoder_y_pred, detr_y_pred)
    far_plot(
        y, y_pred, number_of_signals, total_sample_length, desired_far, ("Encoder", "DETR"), save_as="encoder_detr_far"
    )
    y = (cnn_y, encoder_y, detr_y)
    y_pred = (cnn_y_pred, encoder_y_pred, detr_y_pred)
    far_plot(
        y, y_pred, number_of_signals, total_sample_length, desired_far, ("CNN", "Encoder", "DETR"), zoom=[False, True, True], save_as="cnn_encoder_detr_far"
    )

    num_mergers = np.isfinite(time).sum(axis=-1)
    cnn_tp = (match(cnn_y_pred, cnn_pred_idx) > cnn_threshold).sum(-1)
    encoder_tp = (match(encoder_y_pred, encoder_pred_idx) > encoder_threshold).sum(-1)
    detr_tp = (match(detr_y_pred, detr_pred_idx) > detr_threshold).sum(-1)
    num_mergers_plot((cnn_tp, encoder_tp), num_mergers, labels=("CNN", "Encoder"), save_as="cnn_encoder_mergers")
    num_mergers_plot((encoder_tp, detr_tp), num_mergers, labels=("Encoder", "DETR"), save_as="encoder_detr_mergers")
    num_mergers_plot((cnn_tp, encoder_tp, detr_tp), num_mergers, 
                     labels=("CNN", "Encoder", "DETR"), save_as="cnn_encoder_detr_mergers")