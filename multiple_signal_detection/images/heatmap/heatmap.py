#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join("/data/swein/Documents")))


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


from ggwd.utils.configreader import ConfigReader
from ggwd.utils.samplegenerator import SampleGenerator


config_reader = ConfigReader("one_signal.ini")   
sample_parameters = config_reader.sample_parameters
td_length  = config_reader.sample_parameters["td_length"]
delta_t = config_reader.sample_parameters["delta_t"]
time = np.arange(int(td_length))*delta_t
sample_length = int(td_length)*delta_t

sample_generator = SampleGenerator("one_signal.ini")   
rng = np.random.default_rng()
np.random.seed(42)
out = sample_generator._generate_sample(rng)

waveform = np.array([out["waveform"]["e1"], out["waveform"]["e2"], out["waveform"]["e3"]])
merger_time = out["parameters"]["waveform_parameters"]["merger_time"]
heatmap = np.zeros(td_length)
for t in merger_time:
    if np.isnan(t): continue
    t *= td_length * delta_t
    sigma_in_seconds = 0.1
    gaussian = np.exp(-np.power((time-t)/sigma_in_seconds, 2) / 2)
    i0 = min(0, int((sigma_in_seconds)/delta_t))
    i1 = max(td_length-1, int((t+3*sigma_in_seconds)/delta_t))
    gaussian[:i0] = 0
    gaussian[i1:] = 0
    heatmap = np.maximum.reduce((heatmap, gaussian))

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(7/2.54, 6/2.54), 
                        height_ratios=(1, 2))

sample_length = time[-1]
x0, x1 = np.min(merger_time)*sample_length, np.max(merger_time)*sample_length
delta = 0.4
mask = np.logical_and(x0-delta<time, time<=x1+delta)
x = time[mask]
y = waveform[:,mask]

max_strain = 1.5E-23
ticks = list()
for i, strain in enumerate(waveform):
    axs[0].plot(time, 3/2*(1-i)*max_strain+strain)
    ticks.append(3/2*(1-i)*max_strain)
for t in merger_time:
    axs[0].axvline(t*sample_length, ls="dotted", lw=0.75, c="k")
axs[0].set_yticks(ticks, ["ET1", "ET2", "ET3"])
axs[0].set_ylim(-1.05*5/2*max_strain, 1.05*5/2*max_strain)

axs[1].plot(x, heatmap[mask], c="tab:gray")

axs[0].set_xlim(x[0], x[-1])    
axs[1].set_ylim(-0.05, 1.05)    
axs[1].legend(frameon=False, loc=1)

axs[1].set_xlabel("Time (s)")
axs[0].set_ylabel("Strain")
axs[1].set_ylabel("Heatmap")

axs[0].tick_params(axis="x", which="both", bottom=False, 
                    labelbottom=False)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig("heatmap_one_signal.png", dpi=300)


config_reader = ConfigReader("two_signals.ini")   
sample_parameters = config_reader.sample_parameters
td_length  = config_reader.sample_parameters["td_length"]
delta_t = config_reader.sample_parameters["delta_t"]
time = np.arange(int(td_length))*delta_t
sample_length = int(td_length)*delta_t

sample_generator = SampleGenerator("two_signals.ini")   
rng = np.random.default_rng()
np.random.seed(42)
out = sample_generator._generate_sample(rng)

waveform = np.array([out["waveform"]["e1"], out["waveform"]["e2"], out["waveform"]["e3"]])
merger_time = out["parameters"]["waveform_parameters"]["merger_time"]
gaussians = np.zeros((np.isfinite(merger_time).sum(),td_length))
heatmap = np.zeros(td_length)
for i, t in enumerate(merger_time):
    if np.isnan(t): continue
    t *= td_length * delta_t
    sigma_in_seconds = 0.1
    gaussian = np.exp(-np.power((time-t)/sigma_in_seconds, 2) / 2)
    i0 = min(0, int((sigma_in_seconds)/delta_t))
    i1 = max(td_length-1, int((t+3*sigma_in_seconds)/delta_t))
    gaussian[:i0] = 0
    gaussian[i1:] = 0
    gaussians[i] = gaussian
    heatmap = np.maximum.reduce((heatmap, gaussian))

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9/2.54, 6/2.54), 
                        height_ratios=(1, 2))

sample_length = time[-1]
x0, x1 = np.min(merger_time)*sample_length, np.max(merger_time)*sample_length
delta = 0.4
mask = np.logical_and(x0-delta<time, time<=x1+delta)
x = time[mask]
y = waveform[:,mask]

max_strain = 1.5E-23
ticks = list()
for i, strain in enumerate(waveform):
    axs[0].plot(time, 3/2*(1-i)*max_strain+strain)
    ticks.append(3/2*(1-i)*max_strain)
for t in merger_time:
    axs[0].axvline(t*sample_length, ls="dotted", lw=0.75, c="k")
axs[0].set_yticks(ticks, ["ET1", "ET2", "ET3"])
axs[0].set_ylim(-1.05*5/2*max_strain, 1.05*5/2*max_strain)

for g in gaussians: axs[1].plot(x, g[mask], c="tab:gray", ls="--")
axs[1].plot(x, heatmap[mask], c="tab:gray")


axs[0].set_xlim(x[0], x[-1])    
axs[1].set_ylim(-0.05, 1.05)    
axs[1].legend(frameon=False, loc=1)

axs[1].set_xlabel("Time (s)")
axs[0].set_ylabel("Strain")
axs[1].set_ylabel("Heatmap")

axs[0].tick_params(axis="x", which="both", bottom=False, 
                    labelbottom=False)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig("heatmap_two_signals.png", dpi=300)


get_ipython().system('jupyter nbconvert --to script --no-prompt heatmap.ipynb')
get_ipython().system('python heatmap.py')




