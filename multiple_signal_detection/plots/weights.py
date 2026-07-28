from glob import glob
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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


def plot_weights(time, waveform, merger_time, weights, max_weight, save_as):

    fig, axs = plt.subplots(nrows=2, figsize=(15/2.54, 5/2.54), 
                            height_ratios=(1, 2))


    sample_length = time[-1]
    max_strain = 1.5E-23
    ticks = list()
    for i, strain in enumerate(waveform):
        axs[0].plot(time, 3/2*(1-i)*max_strain+strain)
        ticks.append(3/2*(1-i)*max_strain)
    for t in merger_time:
        axs[0].axvline(t*sample_length, ls="dotted", lw=0.75, c="k")
    axs[0].set_xlim(time[0], time[-1])    
    axs[0].set_ylim(-1.05*5/2*max_strain, 1.05*5/2*max_strain)
    axs[0].set_yticks(ticks, ["ET1", "ET2", "ET3"])
    axs[0].tick_params(axis="x", which="both", bottom=False, 
                       labelbottom=False)

    img = axs[1].imshow(
        weights, 
        vmin=0, vmax=max_weight, 
        interpolation="none", 
        extent=(time[0],time[-1],time[-1],time[0]), 
        aspect="auto"
    )
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Time (s)")

    fig.tight_layout(rect=[0, 0, 0.87, 1])
    pos0 = axs[0].get_position()
    pos1 = axs[1].get_position()
    bottom = min(pos0.y0, pos1.y0)
    top = max(pos0.y1, pos1.y1)
    height = top - bottom
    cbar_ax = fig.add_axes([0.885, bottom, 0.015, height])
    fig.colorbar(img, cax=cbar_ax, label="Encoder weight")
    fig.subplots_adjust(hspace=0)

    fig.savefig(save_as, dpi=300)
    plt.close(fig)


if __name__ == "__main__":

    data_dir = "/scratch/tmp/swein/ggwd/output/test/300"
    data_files = list(Path(data_dir).glob("*.hdf"))
    data_files = sorted(data_files, key=lambda x: int(x.stem))

    encoder_dir = "/scratch/tmp/swein/ggwd/results/heatmap_out/300"
    encoder_files = list(Path(encoder_dir).glob("*.hdf"))
    encoder_files = sorted(encoder_files, key=lambda x: int(x.stem))

    with h5py.File(data_files[0], "r") as file:
        td_length = int(file["timeseries"].attrs["td_length"])
        sampling_rate = file["timeseries"].attrs["sampling_rate"]
    time = np.arange(td_length)/sampling_rate
    N = 24
    waveform = np.array([read_dataset(data_files, "timeseries/waveforms/e1", max_rows=N), 
                            read_dataset(data_files, "timeseries/waveforms/e2", max_rows=N), 
                            read_dataset(data_files, "timeseries/waveforms/e3", max_rows=N)])
    waveform = np.transpose(waveform, (1, 0, 2))
    merger_time = read_dataset(data_files, "parameters/merger_time", max_rows=N)
    weights = read_dataset(encoder_files, "encoder_weights", max_rows=N)
    max_weight = np.max(weights)

    path = Path(f"/home/s/swein/plots/weights/encoder")
    path.mkdir(parents=True, exist_ok=True) 
    for i in range(N):
        plot_weights(
            time, 
            waveform[i], 
            merger_time[i], 
            weights[i],
            max_weight,
            save_as=path / Path(f"{i}.png")
        )