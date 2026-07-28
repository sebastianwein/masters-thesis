from glob import glob
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from uncertainties import unumpy


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


def sigma(p, n):
    return np.sqrt(p*(1-p)/n)


def roc_score(y, y_pred):

    num_points = 1000
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

    fig, ax = plt.subplots(figsize=(7/2.54, 7/2.54))
    ax.plot(fpr, tpr)
    ax.scatter(fpr_opt, tpr_opt, marker="o", color="tab:red", zorder=10)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.5, c="k")
    ax.text(0.75, 0.25, f"AUROC = {auc:.3f}", ha="center", va="center")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.tight_layout()
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/auroc.png", dpi=300, bbox_inches="tight")

    return threshold_opt

def bar_plot(x, y):
    fig, ax = plt.subplots(figsize=(8/2.54, 5/2.54))
    widths = (x[1:] - x[:-1])
    p = ax.bar(x[:-1], y, widths, align="edge", edgecolor="k")
    labels = [f"{round(100*height)}" for height in y]
    ax.bar_label(p, labels, label_type="center", color="white")
    # for i, _ in enumerate(y):
    #     va = "top" if y[i] > 0.9*max(y) else "bottom"
    #     c = "white" if y[i] > 0.9*max(y) else "k"
    #     ax.text(x[i]+widths[i]/2, y[i], f"{int(100*y[i])}", 
    #             ha="center", va=va, c=c)
    ax.set_ylim(0, 1)
    ax.set_xlim(np.min(x), np.max(x))
    return fig, ax

def snr_plot(tp, fn, snr, num_bins = 10): 
    snr_bins = np.linspace(np.min(snr), np.max(snr), num_bins)
    digitized = np.digitize(snr, snr_bins)
    digitized = np.clip(digitized, 1, num_bins-1) - 1
    ratio = lambda x, y, mask: x[mask].sum() / (x[mask].sum() + y[mask].sum())
    detection_ratio = [ratio(tp, fn, digitized==i) for i in range(num_bins-1)]
    uncertainties = [sigma(p, n) for p, n in zip(detection_ratio, [(digitized==i).sum() for i in range(num_bins-1)])]
    arr = unumpy.uarray(detection_ratio, uncertainties)
    print([f"{x:.1u}" for x in arr])
    fig, ax = bar_plot(snr_bins, detection_ratio)
    ax.set_xlabel("Optimal SNR")
    ax.set_ylabel("Detection ratio")
    fig.tight_layout()
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/snr.png", dpi=300, bbox_inches="tight")

def merger_time_plot(tp, fn, time, num_bins = 10): 
    time_bins = np.linspace(np.min(time), np.max(time), num_bins)
    digitized = np.digitize(time, time_bins)
    digitized = np.clip(digitized, 1, num_bins-1) - 1
    ratio = lambda x, y, mask: x[mask].sum() / (x[mask].sum() + y[mask].sum())
    detection_ratio = [ratio(tp, fn, digitized==i) for i in range(num_bins-1)]
    uncertainties = [sigma(p, n) for p, n in zip(detection_ratio, [(digitized==i).sum() for i in range(num_bins-1)])]
    arr = unumpy.uarray(detection_ratio, uncertainties)
    print([f"{x:.1u}" for x in arr])
    fig, ax = bar_plot(time_bins, detection_ratio)
    ax.set_xlabel("Coalescence time (s)")
    ax.set_ylabel("Detection ratio")
    fig.tight_layout()
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/merger_time.png", dpi=300, bbox_inches="tight")

def mesh_plot(x, y, z, normalize=True):
    fig, ax = plt.subplots(figsize=(8/2.54, 7/2.54))
    clim = (0, 1) if normalize else None
    pc = ax.pcolormesh(x, y, z.T, cmap="YlGnBu", clim=clim)
    cmap = pc.get_cmap()
    cbar = fig.colorbar(pc)
    x_mid = x[:-1] + (x[1:] - x[:-1]) / 2
    y_mid = y[:-1] + (y[1:] - y[:-1]) / 2
    brightness = lambda r, g, b: 0.2126*r + 0.7152*g + 0.0722*b
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            color = "white" if brightness(*cmap(z[i,j])[:-1]) < 0.6 else "k"
            ax.text(x_mid[i], y_mid[j], f"{round(100*z[i,j] if normalize else z[i,j])}", 
                    color=color, ha="center", va="center")
    return fig, ax, cbar

def mass_plot(tp, fn, m1, m2, num_bins = 20): 
    m1_bins = np.linspace(np.min(m1), np.max(m1), num_bins)
    m1_digitized = np.digitize(m1, m1_bins)
    m1_digitized = np.clip(m1_digitized, 1, num_bins-1) - 1
    m2_bins = np.linspace(np.min(m2), np.max(m2), num_bins)
    m2_digitized = np.digitize(m2, m2_bins)
    m2_digitized = np.clip(m2_digitized, 1, num_bins-1) - 1
    ratio = lambda x, y, mask: x[mask].sum() / (x[mask].sum() + y[mask].sum())
    detection_ratio = np.zeros((num_bins-1, num_bins-1))
    total = np.zeros((num_bins-1, num_bins-1))
    for i in range(num_bins-1):
        for j in range(num_bins-1):
            if j > i: continue
            detection_ratio[i,j] = ratio(
                tp, fn, np.logical_and(m1_digitized==i, m2_digitized==j)
            )
            detection_ratio[j,i] = detection_ratio[i,j] 
            total[i,j] = np.logical_and(m1_digitized==i, m2_digitized==j).sum()
            total[j,i] = total[i,j] 

    uncertainties = [[sigma(p, n) for p, n in zip(p_row, n_row)] for p_row, n_row in zip(detection_ratio, total)]
    arr = unumpy.uarray(detection_ratio, uncertainties)
    for row in arr: print([f"{x:.1u}" for x in row])

    fig, ax, cbar = mesh_plot(m1_bins, m2_bins, detection_ratio)
    ax.set_xlabel(r"Primary mass $m_1$ ($\mathrm{M}_\odot$)")
    ax.set_ylabel(r"Secondary mass $m_2$ ($\mathrm{M}_\odot$)")
    cbar.set_label("Detection ratio")
    fig.tight_layout()
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/mass.png", dpi=300, bbox_inches="tight")
    fig, ax, cbar = mesh_plot(m1_bins, m2_bins, total, normalize=False)
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/mass_samples.png", dpi=300, bbox_inches="tight")

def spin_plot(tp, fn, spin1z, spin2z, num_bins = 20): 
    spin1z_bins = np.linspace(np.min(spin1z), np.max(spin1z), num_bins)
    spin1z_digitized = np.digitize(spin1z, spin1z_bins)
    spin1z_digitized = np.clip(spin1z_digitized, 1, num_bins-1) - 1
    spin2z_bins = np.linspace(np.min(spin2z), np.max(spin2z), num_bins)
    spin2z_digitized = np.digitize(spin2z, spin2z_bins)
    spin2z_digitized = np.clip(spin2z_digitized, 1, num_bins-1) - 1
    ratio = lambda x, y, mask: x[mask].sum() / (x[mask].sum() + y[mask].sum())
    detection_ratio = np.zeros((num_bins-1, num_bins-1))
    total = np.zeros((num_bins-1, num_bins-1))
    for i in range(num_bins-1):
        for j in range(num_bins-1):
            detection_ratio[i,j] = ratio(
                tp, fn, np.logical_and(spin1z_digitized==i, spin2z_digitized==j)
            )
            total[i,j] = np.logical_and(spin1z_digitized==i, spin2z_digitized==j).sum()

    uncertainties = [[sigma(p, n) for p, n in zip(p_row, n_row)] for p_row, n_row in zip(detection_ratio, total)]
    arr = unumpy.uarray(detection_ratio, uncertainties)
    for row in arr: print([f"{x:.2u}" for x in row])

    fig, ax, cbar = mesh_plot(spin1z_bins, spin2z_bins, detection_ratio)
    ax.set_xlabel("Primary spin $\chi_z^1$")
    ax.set_ylabel("Secondary spin $\chi_z^2$")
    cbar.set_label("Detection ratio")
    fig.tight_layout()
    fig.savefig("/home/s/swein/cnn_vs_vit/plots/spins.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    metrics_dir = "/scratch/tmp/swein/ggwd/results/binary_out/5900/vit/"
    metrics_files = list(Path(metrics_dir).glob("*.hdf"))
    metrics_files = sorted(metrics_files, key=lambda x: int(x.stem))

    for i, path in enumerate(metrics_files):
        with h5py.File(path, "r") as file:
            y_batch = np.bool(np.array(file["ground_truth"]))
            y_pred_batch = np.array(file["predicitions"])
        if i == 0:
            y = y_batch
            y_pred = y_pred_batch
        else:
            y = np.concatenate((y, y_batch))
            y_pred = np.concatenate((y_pred, y_pred_batch))

    threshold = roc_score(y, y_pred)
    y_pred = y_pred > threshold
    tp = (y_pred[y] == 1)
    fn = (y_pred[y] == 0)
    fp = (y_pred[np.logical_not(y)] == 1)
    tn = (y_pred[np.logical_not(y)] == 0)

    data_dir = "/scratch/tmp/swein/ggwd/output/binary/5900"
    data_files = list(Path(data_dir).glob("*.hdf"))
    data_files = sorted(data_files, key=lambda x: int(x.stem))

    mask_nan = lambda x: x[np.logical_not(np.isnan(x))]

    for i, path in enumerate(data_files):
        with h5py.File(path, "r") as file:
            snr_batch = mask_nan(np.array(file["parameters/nomf_snr"]).flatten())
            merger_time_batch = mask_nan(np.array(file["parameters/merger_time"]).flatten())*64
            m1_batch = mask_nan(np.array(file["parameters/mass1"]).flatten())
            m2_batch = mask_nan(np.array(file["parameters/mass2"]).flatten())
            spin1z_batch = mask_nan(np.array(file["parameters/spin1z"]).flatten())
            spin2z_batch = mask_nan(np.array(file["parameters/spin2z"]).flatten())
        if i == 0:
            snr = snr_batch
            merger_time = merger_time_batch
            m1 = m1_batch
            m2 = m2_batch
            spin1z = spin1z_batch
            spin2z = spin2z_batch
        else:
            snr = np.concatenate((snr, snr_batch))
            merger_time = np.concatenate((merger_time, merger_time_batch))
            m1 = np.concatenate((m1, m1_batch))
            m2 = np.concatenate((m2, m2_batch))
            spin1z = np.concatenate((spin1z, spin1z_batch))
            spin2z = np.concatenate((spin2z, spin2z_batch))

    snr_plot(tp, fn, snr)
    merger_time_plot(tp, fn, merger_time)
    mass_plot(tp, fn, m1, m2, num_bins=8)
    spin_plot(tp, fn, spin1z, spin2z, num_bins=8)

