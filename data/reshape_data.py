import torch
from tqdm import tqdm
import numpy as np


def generate_train_test1(V_data, train_data_ratio=0.7):
    """ Given input data and the train/data ratio, outputs batched train and test data. If the ratio is a one decimal number,
        the function output a total of 10 batches. If the ratio is a two decimal number, it outputs 100 batches."""

    # Get number of decimals, determine number of batches
    if train_data_ratio * 10 - int(train_data_ratio * 10) < 1e-10:
        n_batches = 10
    elif train_data_ratio * 100 - int(train_data_ratio * 100) < 1e-10:
        n_batches = 100
    else:
        raise ValueError('train/data ratio must be a one or two decimal number.')

    N_V, T = V_data.shape
    T_batch = int(T / n_batches)
    V_train = torch.zeros(N_V, T_batch, int(n_batches * train_data_ratio), dtype=torch.float32)
    V_test = torch.zeros(N_V, T_batch, int(n_batches * (1 - train_data_ratio)), dtype=torch.float32)
    idt = torch.randperm(n_batches) * T_batch
    j = 0

    # create new data
    for i, idx in enumerate(idt):
        if i / n_batches < train_data_ratio:
            V_train[:, :, i] = V_data[:, idx:idx + T_batch]
        else:
            V_test[:, :, j] = V_data[:, idx:idx + T_batch]
            j += 1

    return V_train, V_test


def make_voxel_xyz(n, spikes, xyz, mode=1, fraction=0.5):
    n = n + 1  # number of voxels
    x = torch.linspace(torch.min(xyz[:, 0]), torch.max(xyz[:, 0]), n)
    y = torch.linspace(torch.min(xyz[:, 1]), torch.max(xyz[:, 1]), n)
    z = torch.linspace(torch.min(xyz[:, 2]), torch.max(xyz[:, 2]), n)

    voxel_xyz = torch.zeros((n - 1) ** 3, 3)
    voxel_spike = torch.zeros((n - 1) ** 3, spikes.shape[1])
    i = 0
    for ix in tqdm(range(n - 1)):
        for iy in range(n - 1):
            for iz in range(n - 1):
                condition = ((xyz[:, 0] > x[ix]) & (xyz[:, 0] < x[ix + 1]) & (xyz[:, 1] > y[iy]) &
                             (xyz[:, 1] < y[iy + 1]) & (xyz[:, 2] > z[iz]) & (xyz[:, 2] < z[iz + 1]))

                if torch.sum(condition) == 0:
                    continue
                V = spikes[condition, :]
                if mode == 1:
                    voxel_spike[i, :] = torch.mean(V, 0)
                if mode == 2:
                    voxel_spike[i, :] = torch.max(V, 0)[0]
                if mode == 3:
                    voxel_spike[i, :] = torch.mean(
                        torch.sort(V, dim=0, descending=True)[0][:int(np.ceil(fraction * V.shape[0])), :], 0)

                voxel_xyz[i, 0] = x[ix]
                voxel_xyz[i, 1] = y[iy]
                voxel_xyz[i, 2] = z[iz]
                i += 1

    condition = ((voxel_xyz[:, 0] > 0) & (voxel_xyz[:, 1] > 0) & (voxel_xyz[:, 2] > 0))
    voxel_xyz = voxel_xyz[condition, :]
    voxel_spike = voxel_spike[condition, :]

    return voxel_spike, voxel_xyz


def generate_train_test(V_data, train_data_ratio=0.7, mode=1):
    """ Given input data and the train/data ratio, outputs batched train and test data. If the ratio is a one decimal number,
        the function output a total of 10 batches. If the ratio is a two decimal number, it outputs 100 batches."""

    # Get number of decimals, determine number of batches
    if train_data_ratio * 10 - int(train_data_ratio * 10) < 1e-10:
        n_batches = 10
    elif train_data_ratio * 100 - int(train_data_ratio * 100) < 1e-10:
        n_batches = 100
    else:
        raise ValueError('train/data ratio must be a one or two decimal number.')

    if torch.tensor(V_data.shape).shape[0] == 2:
        N_V, T = V_data.shape
        train_length = int(n_batches * train_data_ratio)
        test_length = n_batches - train_length
        T_batch = int(T / n_batches)
        V_train = torch.zeros([N_V, T_batch, train_length], dtype=torch.float32)
        V_test = torch.zeros([N_V, T_batch, test_length], dtype=torch.float32)

        if mode == 1:
            # create new data
            V_train = torch.zeros([N_V, train_length, T_batch], dtype=torch.float32)
            V_test = torch.zeros([N_V, test_length, T_batch], dtype=torch.float32)
            for i in range(T_batch):
                V_train[:, :, i] = V_data[:, n_batches * i: n_batches * i + train_length]
                V_test[:, :, i] = V_data[:, n_batches * (i + 1) - test_length: n_batches * (i + 1)]

    elif torch.tensor(V_data.shape).shape[0] == 3:
        N_V, T, n_samples = V_data.shape
        train_length = int(n_samples * train_data_ratio)
        test_length = n_samples - train_length
        T_batch = int(T / n_batches)
        V_train = torch.zeros([N_V, T_batch, train_length], dtype=torch.float32)
        V_test = torch.zeros([N_V, T_batch, test_length], dtype=torch.float32)

    if mode == 2:
        idt = torch.randperm(n_batches) * T_batch
        j = 0
        # create new data
        for i, idx in enumerate(idt):
            if i / n_batches < train_data_ratio:
                V_train[:, :, i] = V_data[:, idx:idx + T_batch]
            else:
                V_test[:, :, j] = V_data[:, idx:idx + T_batch]
                j += 1

    return V_train, V_test


def reshape_from_batches(x, mode='stack_batches'):
    if mode == 'stack_batches':
        return torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    elif mode == 'stack_visibles':
        return torch.reshape(x, (x.shape[0] * x.shape[2], x.shape[1]))


def reshape_to_batches(spikes, mini_batch_size=128):
    # reshape to train in batches
    mini_batch_size = mini_batch_size
    nr_batches = (spikes.shape[1] // mini_batch_size)
    spikes = spikes[:, :mini_batch_size * nr_batches]
    V = torch.zeros([spikes.shape[0], mini_batch_size, nr_batches])
    for j in range(nr_batches):
        V[:, :, j] = spikes[:, mini_batch_size * j:mini_batch_size * (j + 1)]


def train_test_split(data, train_batches=80, test_batches=20):
    n_batches = train_batches + test_batches
    batch_size = data.shape[1] // n_batches
    train = torch.zeros(data.shape[0], batch_size, train_batches)
    test = torch.zeros(data.shape[0], batch_size, test_batches)

    batch_index_shuffled = torch.randperm(n_batches)
    i = 0
    for batch in range(train_batches):
        j = batch_index_shuffled[i]
        train[:, :, batch] = data[:, j * batch_size:(j + 1) * batch_size]
        i += 1

    for batch in range(test_batches):
        j = batch_index_shuffled[i]
        test[:, :, batch] = data[:, j * batch_size:(j + 1) * batch_size]
        i += 1

    return train, test


def reshape_list_of_tensors(list_of_tensors):
    reshaped_list_of_tensors = []
    for tensor in list_of_tensors:
        reshaped_list_of_tensors.append(tensor.reshape(tensor.shape[0], tensor.shape[1]*tensor.shape[2]))
    return reshaped_list_of_tensors


def reshape(data, T=None, n_batches=None):
    if n_batches == None:
        N, T, num_samples = data.shape
        data1 = torch.zeros(N, T * num_samples)
        for i in range(num_samples):
            data1[:, T * i:T * (i + 1)] = data[:, :, i]

    elif n_batches and T is not None:
        N, _ = data.shape
        data1 = torch.zeros(N, T, n_batches)
        for i in range(n_batches):
            data1[:, :, i] = data[:, T * i:T * (i + 1)]
    else:
        raise ValueError('Specify n_batches and T')

    return data1

def resample(data, sr, mode=1):
    '''
    :param data: original data
    :param sr: sampling rate
    :param mode: =1 take the mean, =2 take instance value
    :return: downsampled data
    '''

    N_V, T, n_batches = data.shape
    data = np.array(data)

    # make sure that the modulus(T/sr) = 0
    if T % sr != 0:
        data = data[:, :int(T / sr), :]
    s = int(T / sr)
    data_nsr = np.zeros([N_V, s, n_batches])

    for t in range(s):
        if mode == 1:
            data_nsr[:, t, :] = np.mean(data[:, sr*t:sr*(t + 1), :], axis=1)
        elif mode == 2:
            data_nsr[:, t, :] = data[:, sr*t, :]

    if mode == 1: #need to be carefull for taking the mean of even number >=0.5 will give an additional bias
        data_nsr.ravel()[data_nsr.ravel() >= 0.5] = 1.0
        data_nsr.ravel()[data_nsr.ravel() < 0.5] = 0.0


    return torch.tensor(data_nsr, dtype=torch.float)
