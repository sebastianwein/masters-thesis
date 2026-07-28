from glob import glob
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from uncertainties import unumpy

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


def deltat_stacked(tp, save_as): 
    found_both = np.array([(np.logical_and(x[:,0]==True, x[:,1]==True)).sum()/len(x) for x in tp])
    found_a = np.array([(np.logical_and(x[:,0]==True, x[:,1]==False)).sum()/len(x) for x in tp])
    found_b = np.array([(np.logical_and(x[:,0]==False, x[:,1]==True)).sum()/len(x) for x in tp])
    missed_both = np.array([(np.logical_and(x[:,0]==False, x[:,1]==False)).sum()/len(x) for x in tp])
    fig, ax = plt.subplots(figsize=(11/2.54, 6/2.54))
    if len(tp) == 6: 
        separation = np.array(["30", "10", "3", "1", "0.3", "0.1"])
        width = np.array([1, 1, 1, 1, 1, 1])
    elif len(tp) == 8: 
        separation = np.array(["30", "10", "3", "1", "0.6", "0.3", "0.2", "0.1"])
        width = np.array([1, 1, 1, 3/5, 3/5, 3/5, 3/5, 3/5])
    x = np.cumsum((width[:-1]+width[1:])/2)
    x = np.insert(x, 0, 0)
    kwargs = {"width": width, "edgecolor": "k"}
    ax.bar(x, found_b, label="Found B", color="C0", **kwargs)
    ax.bar(x, found_a, bottom=found_b, label="Found A", color="C3", **kwargs)
    ax.bar(x, found_both, bottom=found_a+found_b, label="Found both", color="C2", **kwargs)
    ax.bar(x, missed_both, bottom=found_both+found_a+found_b, label="Missed both", color="tab:gray", **kwargs)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.set_xlim(x[0]-width[0]/2, x[-1]+width[-1]/2)  
    ax.set_xticks(x, separation)
    ax.set_ylim(0, 1)      
    ax.set_xlabel("Signal separation (s)")
    ax.set_ylabel("Stacked ratio")
    fig.tight_layout()
    fig.savefig(f"/home/s/swein/plots/{save_as}.png", dpi=300, bbox_inches="tight")

def deltat_plot(tp, labels, save_as):
    tp1 = np.array([x.sum()/(x.shape[0]*x.shape[1]) for x in tp[0]])
    n = tp[0][0].shape[0]*tp[0][0].shape[1]
    yerr1 = np.sqrt(tp1*(1-tp1) / n)
    arr = unumpy.uarray(tp1, yerr1)
    print([f"{x:.1uS}" for x in arr])
    print(f"{(np.max(arr[:-3])-np.min(arr[:-3])):.1uS}")
    print(f"{np.mean(arr[:-3]):.1uS}")
    tp2 = np.array([x.sum()/(x.shape[0]*x.shape[1]) for x in tp[1]])
    yerr2 = np.sqrt(tp2*(1-tp2) / n)
    arr = unumpy.uarray(tp2, yerr2)
    print([f"{x:.1uS}" for x in arr])
    print(f"{(np.max(arr[:-3])-np.min(arr[:-3])):.1uS}")
    print(f"{np.mean(arr[:-3]):.1uS}")
    fig, ax = plt.subplots(figsize=(12/2.54, 6/2.54))
    ax.set_axisbelow(True)
    # ax.grid(axis="y")
    if len(tp[0]) == 6: 
        separation = np.array(["30", "10", "3", "1", "0.3", "0.1"])
        width = np.array([1, 1, 1, 1, 1, 1])
    elif len(tp[0]) == 8: 
        separation = np.array(["30", "10", "3", "1", "0.6", "0.3", "0.2", "0.1"])
        width = np.array([1, 1, 1, 3/5, 3/5, 3/5, 3/5, 3/5])
    x = np.cumsum(1.3*(width[:-1]+width[1:]))
    x = np.insert(x, 0, 0)
    p = ax.bar(x-width/2, tp1, width, label=labels[0]) # yerr=yerr1, capsize=4, error_kw={"lw": 1})
    # label = [f"{round(100*x)}" for x in tp1]
    # ax.bar_label(p, label, c="white", label_type="center")
    p = ax.bar(x+width/2, tp2, width, label=labels[1]) # yerr=yerr2, capsize=4, error_kw={"lw": 1})
    # label = [f"{round(100*x)}" for x in tp2]
    # ax.bar_label(p, label, c="white", label_type="center")
    lower = np.min([tp1, tp2]) 
    upper = np.max([tp1, tp2])
    delta = upper - lower
    ax.set_ylim(lower-0.2*delta, upper+0.2*delta)
    ax.set_xticks(x, separation)
    ax.legend(frameon=False)
    ax.set_xlabel("Signal separation (s)")
    ax.set_ylabel("Detection ratio")
    fig.tight_layout()
    fig.savefig(f"/home/s/swein/plots/{save_as}.png", 
                dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    N = 32*4096

    data_dir = "/scratch/tmp/swein/ggwd/output/test/300"
    data_files = list(Path(data_dir).glob("*.hdf"))
    data_files = sorted(data_files, key=lambda x: int(x.stem))

    cnn_dir = "/scratch/tmp/swein/ggwd/results/heatmap_no_encoder_out/300"
    cnn_files = list(Path(cnn_dir).glob("*.hdf"))
    cnn_files = sorted(cnn_files, key=lambda x: int(x.stem))

    encoder_dir = "/scratch/tmp/swein/ggwd/results/heatmap_out/300"
    encoder_files = list(Path(encoder_dir).glob("*.hdf"))
    encoder_files = sorted(encoder_files, key=lambda x: int(x.stem))

    detr_dir = "/scratch/tmp/swein/ggwd/results/detr_out/300"
    detr_files = list(Path(detr_dir).glob("*.hdf"))
    detr_files = sorted(detr_files, key=lambda x: int(x.stem))

    time = read_dataset(data_files, "parameters/merger_time", max_rows=N)
    number_of_signals = np.isfinite(time).sum()
    sample_length = 64
    total_sample_length = len(time)*sample_length
    desired_far = 1/(10*60)
    sigma = 0.1/sample_length
    
    cnn_y_pred = read_dataset(cnn_files, "y_pred", max_rows=N)
    cnn_t_pred = read_dataset(cnn_files, "t_pred", max_rows=N)
    cnn_y, _, _ = greedy_match(cnn_y_pred, cnn_t_pred, time, sigma)
    cnn_threshold = compute_far(cnn_y, cnn_y_pred, number_of_signals, total_sample_length, desired_far)[2]

    encoder_y_pred = read_dataset(encoder_files, "y_pred", max_rows=N)
    encoder_t_pred = read_dataset(encoder_files, "t_pred", max_rows=N)
    encoder_y, _, _ = greedy_match(encoder_y_pred, encoder_t_pred, time, sigma)
    encoder_threshold = compute_far(encoder_y, encoder_y_pred, number_of_signals, total_sample_length, desired_far)[2]

    detr_y_pred = read_dataset(detr_files, "cls_pred", max_rows=N)
    detr_t_pred = read_dataset(detr_files, "time_pred", max_rows=N)
    detr_y, _, _ = greedy_match(detr_y_pred, detr_t_pred, time, sigma)
    detr_threshold = compute_far(detr_y, detr_y_pred, number_of_signals, total_sample_length, desired_far)[2]

    cnn_tp, encoder_tp, detr_tp = list(), list(), list()

    for seed in [400, 500, 600, 700, 750, 800, 850, 900]:

        print(seed)

        data_dir = f"/scratch/tmp/swein/ggwd/output/test/{seed}"
        data_files = list(Path(data_dir).glob("*.hdf"))
        data_files = sorted(data_files, key=lambda x: int(x.stem))

        time = read_dataset(data_files, "parameters/merger_time")

        cnn_dir = f"/scratch/tmp/swein/ggwd/results/heatmap_no_encoder_out/{seed}"
        cnn_files = list(Path(cnn_dir).glob("*.hdf"))
        cnn_files = sorted(cnn_files, key=lambda x: int(x.stem))

        encoder_dir = f"/scratch/tmp/swein/ggwd/results/heatmap_out/{seed}"
        encoder_files = list(Path(encoder_dir).glob("*.hdf"))
        encoder_files = sorted(encoder_files, key=lambda x: int(x.stem))

        detr_dir = f"/scratch/tmp/swein/ggwd/results/detr_out/{seed}"
        detr_files = list(Path(detr_dir).glob("*.hdf"))
        detr_files = sorted(detr_files, key=lambda x: int(x.stem))

        cnn_y_pred = read_dataset(cnn_files, "y_pred")
        cnn_t_pred = read_dataset(cnn_files, "t_pred")
        cnn_y, cnn_signal_idx, cnn_pred_idx = greedy_match(cnn_y_pred, cnn_t_pred, time, sigma)
        cnn_tp.append(match(cnn_y_pred, cnn_pred_idx) > cnn_threshold)

        encoder_y_pred = read_dataset(encoder_files, "y_pred")
        encoder_t_pred = read_dataset(encoder_files, "t_pred")
        encoder_y, encoder_signal_idx, encoder_pred_idx = greedy_match(encoder_y_pred, encoder_t_pred, time, sigma)
        encoder_tp.append(match(encoder_y_pred, encoder_pred_idx) > encoder_threshold)

        detr_y_pred = read_dataset(detr_files, "cls_pred")
        detr_t_pred = read_dataset(detr_files, "time_pred")
        detr_y, detr_signal_idx, detr_pred_idx = greedy_match(detr_y_pred, detr_t_pred, time, sigma)
        detr_tp.append(match(detr_y_pred, detr_pred_idx) > detr_threshold)

    deltat_stacked(cnn_tp, "cnn_stacked")
    deltat_stacked(encoder_tp, "encoder_stacked")
    deltat_stacked(detr_tp, "detr_stacked")

    deltat_plot((cnn_tp, encoder_tp), ("CNN", "Encoder"), save_as="cnn_encoder_deltat")
    deltat_plot((encoder_tp, detr_tp), ("Encoder", "DETR"), save_as="encoder_detr_deltat")