"""
Always give an ax as input and return it.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils.funcs import get_param_history
from matplotlib.colors import LinearSegmentedColormap


def mean_std_parameters_history(parameter_history, axes=None):
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    W, U, b_H, b_V, b_init = get_param_history(parameter_history)
    epochs = np.arange(W.shape[0])
    params = [W, U, b_H, b_V, b_init]
    for param in params:
        axes[0].plot(epochs, torch.mean(param, (1, 2)))
        axes[1].plot(epochs, torch.std(param, (1, 2)))
    axes[0].legend(['W', 'U', 'b_H', 'b_V', 'b_init'])
    axes[0].set_ylabel('Means', fontsize=15)
    axes[1].set_ylabel('STDs', fontsize=15)
    axes[1].set_xlabel('epochs', fontsize=15)
    plt.tight_layout()

    return axes


def raster_plot(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    colors = ['white', 'black']
    cmap = LinearSegmentedColormap.from_list('', colors, 2)
    sns.heatmap(data, cbar=False, cmap=cmap, vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('time', fontsize=12)
    ax.set_ylabel('# neuron', fontsize=12)
    return ax
