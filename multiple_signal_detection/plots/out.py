from adjustText import adjust_text
from glob import glob
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import interpolate

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


tol_bright = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

tol_high_contrast = [
    "#004488",  # blue
    "#DDAA33",  # yellow
    "#BB5566",  # red
]

tol_vibrant = [
    "#0077BB",  # blue
    "#33BBEE",  # cyan
    "#009988",  # teal
    "#EE7733",  # orange
    "#CC3311",  # red
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
]

tol_muted = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#DDCC77",  # sand
    "#CC6677",  # rose
    "#882255",  # wine
    "#AA4499",  # purple
    "#DDDDDD",  # pale grey
]

tol_medium_contrast = [
    "#6699CC",  # light blue
    "#004488",  # dark blue
    "#EECC66",  # light yellow
    "#994455",  # dark red
    "#997700",  # dark yellow
    "#EE99AA",  # light red
]

tol_pale = [
    "#BBCCEE",  # blue
    "#CCEEFF",  # cyan
    "#CCDDAA",  # green
    "#EEEEBB",  # yellow
    "#FFCCCC",  # red
    "#DDDDDD",  # grey
]

tol_dark = [
    "#222255",  # blue
    "#225555",  # cyan
    "#225522",  # green
    "#666633",  # yellow
    "#663333",  # red
    "#555555",  # grey
]

tol_light = [
    "#77AADD",  # blue
    "#99DDFF",  # cyan
    "#44BB99",  # mint
    "#BBCC33",  # pear
    "#AAAA00",  # olive
    "#EEDD88",  # yellow
    "#EE8866",  # orange
    "#FFAABB",  # pink
    "#DDDDDD",  # pale grey
]


def plot_heatmap(time, waveform, merger_time, ground_truth, heatmap, queries, y, threshold, xlim, save_as,  width=15):
        
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(width/2.54, 5/2.54), 
                            height_ratios=(1, 2))
    
    sample_length = time[-1]
    x0, x1 = np.min(merger_time)*sample_length-0.3, np.max(merger_time)*sample_length+0.3
    delta = 0.1*(x1-x0)
    mask = np.logical_and(x0-delta<time, time<=x1+delta) if xlim is True else slice(None)
    time = time[mask]
    waveform = waveform[:,mask]
    
    max_strain = 1.5E-23
    ticks = list()
    for i, strain in enumerate(waveform):
        axs[0].plot(time, 3/2*(1-i)*max_strain+strain)
        ticks.append(3/2*(1-i)*max_strain)
    for t in merger_time:
        axs[0].axvline(t*sample_length, ls="dotted", lw=0.75, c="k")
    axs[0].set_yticks(ticks, ["ET1", "ET2", "ET3"])
    axs[0].set_ylim(-1.05*5/2*max_strain, 1.05*5/2*max_strain)

    if ground_truth is not None: axs[1].plot(time, ground_truth[mask], label="Ground truth", c="k")
    x = np.linspace(0, 1, len(heatmap), endpoint=False)*sample_length
    axs[1].plot(x, heatmap, label="Prediction", c="tab:gray")
    f = interpolate.interp1d(x, heatmap)
    avoid_x = [np.linspace(np.min(x), np.max(x), 4*len(heatmap))]
    avoid_y = [f(avoid_x)[0]]
    texts = list()
    for i, _ in enumerate(queries): 
        if np.isnan(queries[i,0]) or queries[i,0] < threshold: continue
        c = "tab:red" if y[i] == 0 else "tab:green"   
        axs[1].axvline(queries[i,1]*sample_length, ls="--", lw=1, c=c)
        text = axs[1].text(queries[i,1]*sample_length, 1, f"{100*queries[i,0]:.1f}%", va="top", ha="center")
        texts.append(text)


    axs[0].set_xlim(time[0], time[-1])    
    axs[1].set_ylim(-0.05, 1.05)    

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Heatmap")

    axs[0].tick_params(axis="x", which="both", bottom=False, 
                       labelbottom=False)
    n = 40
    for p, x in queries:
        if np.isnan(p) or p < threshold: continue          
        avoid_x.append(np.ones(n)*x*sample_length)
        avoid_y.append(np.linspace(*axs[1].get_ylim(), n))
    avoid_x = np.concatenate(avoid_x) if avoid_x else None
    avoid_y = np.concatenate(avoid_y) if avoid_y else None
    adjust_text(texts, x=avoid_x, y=avoid_y, only_move="xy-",force_text=0.3, force_static=0.1,
                arrowprops=dict(arrowstyle="-", color="k", lw=0.5, zorder=10))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(save_as, dpi=300)
    matplotlib.pyplot.close()


def plot_detr(time, waveform, merger_time, queries, weights, ground_truth, threshold, xlim, save_as, width=15):
        
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(width/2.54, 5/2.54), 
                            height_ratios=(1, 2))


    sample_length = time[-1]
    x0, x1 = np.min(merger_time)*sample_length-0.3, np.max(merger_time)*sample_length+0.3
    delta = 0.1*(x1-x0)
    mask = np.logical_and(x0-delta<time, time<=x1+delta) if xlim is True else slice(None)
    time = time[mask]
    waveform = waveform[:,mask]

    max_strain = 1.5E-23
    ticks = list()
    for i, strain in enumerate(waveform):
        axs[0].plot(time, 3/2*(1-i)*max_strain+strain)
        ticks.append(3/2*(1-i)*max_strain)
    for t in merger_time:
        axs[0].axvline(t*sample_length, ls="dotted", lw=0.75, c="k")
    axs[0].set_yticks(ticks, ["ET1", "ET2", "ET3"])
    axs[0].set_ylim(-1.05*5/2*max_strain, 1.05*5/2*max_strain)

    avoid_x, avoid_y = list(), list()
    texts = list()
    colors = iter(tol_bright)
    for i, _ in enumerate(queries): 
        if queries[i,0] < threshold: continue
        x = np.linspace(0, sample_length, weights.shape[-1], endpoint=False)
        axs[1].plot(x, weights[i], c=next(colors), alpha=0.7)
        f = interpolate.interp1d(x, weights[i])
        x = np.linspace(np.min(x), np.max(x), 4*weights.shape[-1])
        avoid_x.append(x)
        avoid_y.append(f(x))
        c = "tab:red" if ground_truth[i] == 0 else "tab:green"   
        axs[1].axvline(queries[i,1]*sample_length, ls="--", lw=1, c=c, zorder=10)
        w = np.array([w for w, p in zip(weights, queries[:,0]) if p >= threshold])
        y = np.max(w)
        text = axs[1].text(queries[i,1]*sample_length, y, f"{100*queries[i,0]:.1f}%", va="top", ha="center")
        texts.append(text)

    axs[0].set_xlim(time[0], time[-1])    

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Decoder weights")

    axs[0].tick_params(axis="x", which="both", bottom=False, 
                       labelbottom=False)
    n = 40
    for p, x in queries:
        if p < threshold: continue          
        avoid_x.append(np.ones(n)*x*sample_length)
        avoid_y.append(np.linspace(*axs[1].get_ylim(), n))
    avoid_x = np.concatenate(avoid_x) if avoid_x else None
    avoid_y = np.concatenate(avoid_y) if avoid_y else None
    adjust_text(texts, x=avoid_x, y=avoid_y, only_move="xy-",force_text=0.3, force_static=0.1,
                arrowprops=dict(arrowstyle="-", color="k", lw=0.5, zorder=10))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(save_as, dpi=300)
    matplotlib.pyplot.close()


if __name__ == "__main__":

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

    time = read_dataset(data_files, "parameters/merger_time")
    number_of_signals = np.isfinite(time).sum()
    with h5py.File(data_files[0], "r") as file:
        sample_length = file["timeseries"].attrs["sample_length"]
    total_sample_length = len(time)*sample_length
    desired_far = 1/(10*60)
    sigma = 0.1/sample_length
    
    cnn_y_pred = read_dataset(cnn_files, "y_pred")
    cnn_t_pred = read_dataset(cnn_files, "t_pred")
    cnn_y, _, _ = greedy_match(cnn_y_pred, cnn_t_pred, time, sigma)
    cnn_threshold = compute_far(cnn_y, cnn_y_pred, number_of_signals, total_sample_length, desired_far)[2]

    encoder_y_pred = read_dataset(encoder_files, "y_pred")
    encoder_t_pred = read_dataset(encoder_files, "t_pred")
    encoder_y, _, _ = greedy_match(encoder_y_pred, encoder_t_pred, time, sigma)
    encoder_threshold = compute_far(encoder_y, encoder_y_pred, number_of_signals, total_sample_length, desired_far)[2]

    detr_y_pred = read_dataset(detr_files, "cls_pred")
    detr_t_pred = read_dataset(detr_files, "time_pred")
    detr_y, _, _ = greedy_match(detr_y_pred, detr_t_pred, time, sigma)
    detr_threshold = compute_far(detr_y, detr_y_pred, number_of_signals, total_sample_length, desired_far)[2]

    N = 24
    for seed in [300, 400, 500, 600, 700, 800, 900]:
        print(seed)

        data_dir = f"/scratch/tmp/swein/ggwd/output/test/{seed}"
        data_files = list(Path(data_dir).glob("*.hdf"))
        data_files = sorted(data_files, key=lambda x: int(x.stem))

        with h5py.File(data_files[0], "r") as file:
            td_length = int(file["timeseries"].attrs["td_length"])
            sampling_rate = file["timeseries"].attrs["sampling_rate"]
        time = np.arange(td_length)/sampling_rate
        waveform = np.array([read_dataset(data_files, "timeseries/waveforms/e1", max_rows=N), 
                             read_dataset(data_files, "timeseries/waveforms/e2", max_rows=N), 
                             read_dataset(data_files, "timeseries/waveforms/e3", max_rows=N)])
        waveform = np.transpose(waveform, (1, 0, 2))
        merger_time = read_dataset(data_files, "parameters/merger_time", max_rows=N)
        ground_truth = np.zeros((N, td_length))
        for i in range(N):
            for t in merger_time[i]:
                if np.isnan(t): continue
                t *= td_length / sampling_rate
                sigma_in_seconds = 0.1
                gaussian = np.exp(-np.power((time-t)/sigma_in_seconds, 2) / 2)
                i0 = min(0, int((sigma_in_seconds)*sampling_rate))
                i1 = max(td_length-1, int((t+3*sigma_in_seconds)*sampling_rate))
                gaussian[:i0] = 0
                gaussian[i1:] = 0
                ground_truth[i] = np.maximum.reduce((ground_truth[i], gaussian))

        cnn_dir = f"/scratch/tmp/swein/ggwd/results/heatmap_no_encoder_out/{seed}"
        cnn_files = list(Path(cnn_dir).glob("*.hdf"))
        cnn_files = sorted(cnn_files, key=lambda x: int(x.stem))

        heatmap = read_dataset(cnn_files, "heatmaps", max_rows=N)
        cls_pred = read_dataset(cnn_files, "y_pred", max_rows=N)
        time_pred = read_dataset(cnn_files, "t_pred", max_rows=N)
        queries = np.transpose(np.array([cls_pred, time_pred]), (1, 2, 0))
        y, _, _ = greedy_match(cls_pred, time_pred, merger_time, sigma)
        y = y == 1

        path = Path(f"/home/s/swein/plots/out/{seed}/cnn")
        path.mkdir(parents=True, exist_ok=True) 
        for i in range(N):
            plot_heatmap(
                time, 
                waveform[i], 
                merger_time[i], 
                None, 
                heatmap[i], 
                queries[i], 
                y[i], 
                encoder_threshold, 
                xlim=False, 
                save_as=path / Path(f"{i}.png"),
                width=15
            )
            if seed > 300:
                plot_heatmap(
                    time, 
                    waveform[i], 
                    merger_time[i], 
                    ground_truth[i], 
                    heatmap[i], 
                    queries[i], 
                    y[i], 
                    encoder_threshold, 
                    xlim=True, 
                    save_as=path / Path(f"{i}_zoom.png"),
                    width=7
                )

        encoder_dir = f"/scratch/tmp/swein/ggwd/results/heatmap_out/{seed}"
        encoder_files = list(Path(cnn_dir).glob("*.hdf"))
        encoder_files = sorted(cnn_files, key=lambda x: int(x.stem))

        heatmap = read_dataset(encoder_files, "heatmaps", max_rows=N)
        cls_pred = read_dataset(encoder_files, "y_pred", max_rows=N)
        time_pred = read_dataset(encoder_files, "t_pred", max_rows=N)
        queries = np.transpose(np.array([cls_pred, time_pred]), (1, 2, 0))
        y, _, _ = greedy_match(cls_pred, time_pred, merger_time, sigma)
        y = y == 1

        path = Path(f"/home/s/swein/plots/out/{seed}/encoder")
        path.mkdir(parents=True, exist_ok=True) 
        for i in range(N):
            plot_heatmap(
                time, 
                waveform[i], 
                merger_time[i], 
                None, 
                heatmap[i], 
                queries[i], 
                y[i], 
                encoder_threshold, 
                xlim=False, 
                save_as=path / Path(f"{i}.png"),
                width=15
            )
            if seed > 300:
                plot_heatmap(
                    time, 
                    waveform[i], 
                    merger_time[i], 
                    ground_truth[i], 
                    heatmap[i], 
                    queries[i], 
                    y[i], 
                    encoder_threshold, 
                    xlim=True, 
                    save_as=path / Path(f"{i}_zoom.png"),
                    width=7
                )

        detr_dir = f"/scratch/tmp/swein/ggwd/results/detr_out/{seed}"
        detr_files = list(Path(detr_dir).glob("*.hdf"))
        detr_files = sorted(detr_files, key=lambda x: int(x.stem))

        cls_pred = read_dataset(detr_files, "cls_pred", max_rows=N)
        time_pred = read_dataset(detr_files, "time_pred", max_rows=N)
        queries = np.transpose(np.array([cls_pred, time_pred]), (1, 2, 0))
        weights = read_dataset(detr_files, "decoder_weights", max_rows=N)
        y, _, _ = greedy_match(cls_pred, time_pred, merger_time, sigma)
        y = y == 1

        path = Path(f"/home/s/swein/plots/out/{seed}/detr")
        path.mkdir(parents=True, exist_ok=True) 
        for i in range(N):
            plot_detr(
                time, 
                waveform[i], 
                merger_time[i], 
                queries[i], 
                weights[i], 
                y[i], 
                detr_threshold, 
                xlim=False, 
                save_as=path / Path(f"{i}.png"),
                width=15
            )
            if seed > 300:
                plot_detr(
                    time, 
                    waveform[i], 
                    merger_time[i], 
                    queries[i], 
                    weights[i], 
                    y[i], 
                    detr_threshold, 
                    xlim=True, 
                    save_as=path / Path(f"{i}_zoom.png"),
                    width=7
                )