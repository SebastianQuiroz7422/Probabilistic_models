import h5py
import numpy as np
import os
import torch
import sys
from boltzmann_machines.data.reshape_data import train_test_split


def load_data(data_path=None):
    if data_path is None:
        path = os.path.dirname(os.getcwd())
        sys.path.append(path)
        data_obj = h5py.File(path + '/data/real_zebrafish_data/subject_1_reconv_spikes.h5', 'r')
    else:
        data_obj = h5py.File(data_path, 'r')
    behavior = np.array(data_obj['Data']['behavior'])
    coordinates = np.array(data_obj['Data']['coords'])
    df = np.array(data_obj['Data']['df'])
    spikes = np.array(data_obj['Data']['spikes'])
    stimulus = np.array(data_obj['Data']['stimulus'])

    return spikes, behavior, coordinates, df, stimulus


def load_data_thijs():
    path = os.path.dirname(os.getcwd())
    sys.path.append(path)

    data_obj = h5py.File(path + '/data/real_zebrafish_data/thijs.h5', 'r')
    spikes = np.transpose(np.array(data_obj['Data']['Brain']['Analysis']['ThresholdedSpikes']))
    coordinates = np.transpose(np.array(data_obj['Data']['Brain']['Coordinates']))
    times = np.array(data_obj['Data']['Brain']['Times'])

    return spikes, coordinates, times


def get_split_data(N_V, train_batches=80, test_batches=20):
    spikes, behavior, coordinates, df, stimulus = load_data('/mnt/data/zebrafish/chen2018/subject_1/Deconvolved/subject_1_reconv_spikes.h5')

    # sort spikes by ascending firing rate
    firing_rates = np.mean(spikes, 1)
    sort_idx = np.argsort(firing_rates)[::-1]
    firing_rates_sorted = firing_rates[sort_idx]
    data = spikes[sort_idx, :] > .15
    data = torch.tensor(data, dtype=torch.float)

    # split in 80 train batches and 20 test batches
    return train_test_split(data[:N_V, :], train_batches=train_batches, test_batches=test_batches)