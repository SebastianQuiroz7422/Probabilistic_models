import torch
import numpy as np
from scipy import signal
import random


def poisson_drawn_data(neurons_per_pop=50,
                       n_pop=6,
                       n_batches=300,
                       T=100,
                       inh=None,
                       exc=None):
    """
    Generated artificial data poisson drawn from random sinusoids. Connections are based on removing or adding spikes
    dependent on connection and firing rate of connected population.
    """

    if inh is None:
        inh = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
    if exc is None:
        exc = [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]

    ######## Defining coordinate system ########
    rads = torch.linspace(0, 2 * torch.pi, n_pop + 1)
    mean_locations_pop = torch.zeros(n_pop, 2)
    coordinates = torch.zeros(neurons_per_pop * n_pop, 2)
    for i in range(n_pop):
        mean_locations_pop[i, :] = torch.tensor([torch.cos(rads[i]), torch.sin(rads[i])])
        coordinates[neurons_per_pop * i:neurons_per_pop * (i + 1), :] = 0.15 * torch.randn(neurons_per_pop, 2) + \
                                                                        mean_locations_pop[i]

    ######## Start creating data ########
    data = torch.zeros(neurons_per_pop * n_pop, T, n_batches)
    for batch in range(n_batches):

        ######## Creating random input currents and mother trains ########
        t = np.linspace(0, 10 * np.pi, T)
        fr = np.zeros((n_pop, T))
        mother = np.zeros((n_pop, T))
        for pop in range(n_pop):
            u = np.random.rand()
            phase = np.random.randn()
            amp = .1 * np.random.rand()
            shift = .3 * np.random.rand()
            fr[pop, :] = amp * np.sin(phase * (t + 2 * np.pi * u)) + shift
            while np.min(fr[pop, :]) < 0:
                u = np.random.rand()
                phase = np.random.randn()
                amp = .1 * np.random.rand()
                shift = .3 * np.random.rand()
                fr[pop, :] = amp * np.sin(phase * (t + 2 * np.pi * u)) + shift
            mother[pop, :] = np.random.poisson(fr[pop, :])

        # empty data array
        spikes = np.zeros((neurons_per_pop * n_pop, T))

        # Excitatory and inhibitory connections
        for pop in range(n_pop):
            delete_spikes = np.roll(np.sum(fr[inh[pop], :], 0), 1) * np.ones((neurons_per_pop, T)) >= np.random.uniform(
                0, 1, size=(neurons_per_pop, T))
            noise = np.random.poisson(np.roll(np.sum(fr[exc[pop], :], 0), 1), (neurons_per_pop, T))
            spikes[pop * neurons_per_pop:(pop + 1) * neurons_per_pop, :] = np.tile(mother[pop, :], (
            neurons_per_pop, 1)) - delete_spikes + noise
        spikes[spikes < 0] = 0
        spikes[spikes > 1] = 1

        data[:, :, batch] = torch.tensor(spikes)
        return data

def create_complex_artificial_data(n_neurons=900,
                                   t_max=2500,
                                   n_populations=9,
                                   mean_firing_rate=0.1,
                                   population_correlations=None,
                                   neuron_population_correlation=None,
                                   time_shifts=None,
                                   permute=True
                                   ):
    """
    Create surrogate data of 9 neuron populations with different correlations:
    population 1: no correlation, random spiking
    population 2 - 9: high inter-neuron correlation (0.9)
    population 2: no correlation to other populations
    population 3 - 9: these populations are spatially correlated (.5, .5, .8, .8, .8, .8, .8)
    population 5 - 9: have additional time shifted spikes, which makes them temporally correlated

    These correlations can be adjusted:
        - population_correlations: defines the correlations between populations
        - neuron_population_correlation: defines inter-neuron correlation for every population
        - time-shifts: gives a population a certain lag
        - permute: if true, shuffle neuron indexes

    returns:
        - data: spiking data of all neurons
        - coordinates: brain-space coordinates of these neurons
        - population_idx: np.array of np.arrays of population neuron indexes
    """

    if time_shifts is None:
        time_shifts = [0, 0, 0, 0, 0, -2, -1, 1, 2]
    if neuron_population_correlation is None:
        neuron_population_correlation = [0, .7, .7, .7, .7, .7, .7, .7, .7]
    if population_correlations is None:
        population_correlations = [0, 0, .5, .5, .8, .8, .8, .8, .8]

    n_neurons_pop = int(n_neurons / n_populations)

    spikes = np.zeros((n_neurons, t_max))

    # population 1: uncorrelated
    spikes[:n_neurons_pop, :] = np.random.poisson(lam=mean_firing_rate, size=(n_neurons_pop, t_max))

    # population 2 - 9: correlated
    poisson_grandmother_train = np.random.poisson(lam=mean_firing_rate, size=(1, t_max))
    poisson_mother_trains = np.zeros((n_populations, t_max))
    for i in range(n_populations):
        # population dynamics
        delete_spikes = population_correlations[i] * np.ones((1, t_max)) <= np.random.uniform(0, 1, size=(1, t_max))
        noise = np.random.poisson((1 - population_correlations[i]) * mean_firing_rate,
                                  (1, t_max))  # add noise according to corr
        poisson_mother_trains[i] = poisson_grandmother_train - (
                delete_spikes * poisson_grandmother_train) + noise  # delete spikes, add noise
        poisson_mother_trains[i] = np.roll(poisson_mother_trains[i], time_shifts[i])  # add time shifts

        # neuron dynamics
        delete_spikes = neuron_population_correlation[i] * np.ones((n_neurons_pop, t_max)) <= \
                        np.random.uniform(0, 1, size=(n_neurons_pop, t_max))
        noise = np.random.poisson((1 - neuron_population_correlation[i]) * mean_firing_rate, (n_neurons_pop, t_max))
        population_spikes = np.tile(poisson_mother_trains[i], (n_neurons_pop, 1))
        population_spikes = population_spikes - (delete_spikes * population_spikes) + noise

        idx = np.arange(n_neurons_pop) + i * n_neurons_pop
        spikes[idx, :] = population_spikes

    spikes[spikes < 0] = 0
    spikes[spikes > 1] = 1

    # create 2 dimensional brain-space locations for all neuron populations
    mean_locations_pop = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    neuron_coordinates = np.zeros((n_neurons, 2))
    for i in range(n_populations):
        neuron_coordinates[n_neurons_pop * i:n_neurons_pop * (i + 1), :] = 0.25 * np.random.randn(n_neurons_pop, 2) + \
                                                                           mean_locations_pop[i]

    # randomly permute neurons
    populations_idx = np.arange(n_neurons, dtype=int)
    if permute:
        idx = np.random.permutation(n_neurons)
        neuron_coordinates = neuron_coordinates[idx, :]
        populations_idx = populations_idx[np.argsort(idx)]
        spikes = spikes[idx, :]
    populations_idx = populations_idx.reshape(n_populations, n_neurons_pop)

    return spikes, neuron_coordinates, populations_idx


def cai():
    n_neurons = 900
    t_max = 2500
    n_populations = 9
    permute = True
    time_shifts = [0, 0, 0, 0, 0, 0, 1, 2, 3]
    neuron_population_correlation = [0, .8, .8, .8, .8, .8, .8, .8, .8]
    population_correlations = [0, .6, .6, .6, .6, .6, .9, .9, .9]
    mean_firing_rate = 0.1

    n_neurons_pop = int(n_neurons / n_populations)

    spikes = np.zeros((n_neurons, t_max))

    # population 1 - 3: uncorrelated
    spikes[:n_neurons_pop * 3, :] = np.random.poisson(lam=mean_firing_rate, size=(n_neurons_pop * 3, t_max))

    # population 4 - 6: correlated
    poisson_grandmother_train1 = np.random.poisson(lam=mean_firing_rate, size=(1, t_max))
    poisson_grandmother_train2 = np.random.poisson(lam=mean_firing_rate, size=(1, t_max))

    poisson_mother_trains = np.zeros((n_populations, t_max))
    for i in range(3, 9):

        delete_spikes = population_correlations[i] * np.ones((1, t_max)) <= np.random.uniform(0, 1, size=(1, t_max))
        noise = np.random.poisson((1 - population_correlations[i]) * mean_firing_rate, (1, t_max))
        if i <= 5:
            poisson_mother_trains[i] = poisson_grandmother_train1 - (delete_spikes * poisson_grandmother_train1) + noise
        else:
            poisson_mother_trains[i] = poisson_grandmother_train2 - (delete_spikes * poisson_grandmother_train2) + noise
        poisson_mother_trains[i] = np.roll(poisson_mother_trains[i], time_shifts[i])  # add time shifts

        # neuron dynamics
        delete_spikes = neuron_population_correlation[i] * np.ones((n_neurons_pop, t_max)) <= \
                        np.random.uniform(0, 1, size=(n_neurons_pop, t_max))
        noise = np.random.poisson((1 - neuron_population_correlation[i]) * mean_firing_rate, (n_neurons_pop, t_max))
        population_spikes = np.tile(poisson_mother_trains[i], (n_neurons_pop, 1))
        population_spikes = population_spikes - (delete_spikes * population_spikes) + noise

        idx = np.arange(n_neurons_pop) + i * n_neurons_pop
        spikes[idx, :] = population_spikes

    # make binary
    spikes[spikes < 0] = 0
    spikes[spikes > 1] = 1

    # create 2 dimensional brain-space locations for all neuron populations
    mean_locations_pop = [[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]]
    neuron_coordinates = np.zeros((n_neurons, 2))
    for i in range(n_populations):
        neuron_coordinates[n_neurons_pop * i:n_neurons_pop * (i + 1), :] = 0.25 * np.random.randn(n_neurons_pop, 2) + \
                                                                           mean_locations_pop[i]

    # randomly permute neurons
    populations_idx = np.arange(n_neurons, dtype=int)
    if permute:
        idx = np.random.permutation(n_neurons)
        neuron_coordinates = neuron_coordinates[idx, :]
        populations_idx = populations_idx[np.argsort(idx)]
        spikes = spikes[idx, :]
    populations_idx = populations_idx.reshape(n_populations, n_neurons_pop)
    return torch.tensor(spikes), torch.tensor(neuron_coordinates), populations_idx


def create_BB(N_V=16, T=32, n_samples=256, width_vec=[4, 5, 6, 7], velocity_vec=[1, 2], boundary=False, r=2):
    """ Generate 1 dimensional bouncing ball data with or without boundaries, with different ball widths and velocities"""

    data = np.zeros([N_V, T, n_samples])

    for i in range(n_samples):
        if boundary:
            v = random.sample(velocity_vec, 1)[0]
            dt = 1
            x = np.random.randint(r, N_V - r)
            trend = (2 * np.random.randint(0, 2) - 1)
            for t in range(T):
                if x + r > N_V - 1:
                    trend = -1
                elif x - r < 1:
                    trend = 1
                x += trend * v * dt

                data[x - r:x + r, t, i] = 1
        else:
            ff0 = np.zeros(N_V)
            ww = random.sample(width_vec, 1)[0]
            ff0[0:ww] = 1  # width

            vv = random.sample(velocity_vec, 1)[0]  # initial speed, vv>0 so always going right
            for t in range(T):
                ff0 = np.roll(ff0, vv)
                data[:, t, i] = ff0

    return torch.tensor(data, dtype=torch.float)


def generate_mock_data(N_V, T, sampling_frequency=4, average_firing_rate=0.0213, stimulus_period=480 / 4,
                       random_seed=0):
    """ Generate spike train data of N_V neurons for T timesteps with a given sampling frequency (Hz) and avarage firing rate.
        Stimulus period is taken from Misha's data.
        Returns spike trains and time array. """

    # set random seed
    torch.random.manual_seed(random_seed)

    # calculate stimulus frequency
    stimulus_frequency = 1 / stimulus_period

    # create time array
    time_arr = torch.linspace(0, T / sampling_frequency, T)

    # calculate firing probability with triangle wave
    firing_probability = torch.tensor(
        average_firing_rate * (signal.sawtooth(2 * np.pi * stimulus_frequency * time_arr, 0.5) + 1))

    # spike trains
    spikes = torch.rand(N_V, T) < firing_probability.expand(N_V, T)
    spikes = spikes.type(torch.DoubleTensor)

    return spikes, time_arr


def generate_data_fr_dist(N_V, T, spikes_true, true_sampling_rate=4, nr_of_discretized_steps=1000, randperm=True):
    """ This function creates mock data, but with the right firing rate distribution as given by spikes_true data. """

    # sort the spikes according to firing rate
    f_rate_order_true = torch.argsort(torch.sum(spikes_true, 1), descending=True)
    spikes_true = spikes_true[f_rate_order_true, :]

    # create empty data tensor
    data = torch.zeros(N_V, T)

    # loop over number of discretized steps
    for i in range(0, nr_of_discretized_steps):
        # an indexer
        idx = int(spikes_true.shape[0] / nr_of_discretized_steps)

        # calculate mean firing rate of spikes_true batch
        mean_firing_rate_discretized = torch.mean(spikes_true[i * idx:(i + 1) * idx, :])

        # use mean firing rate to generate mock data
        data[int(i * N_V / nr_of_discretized_steps):int((i + 1) * N_V / nr_of_discretized_steps), :], time_arr = \
            generate_mock_data(int(N_V / nr_of_discretized_steps), T,
                               average_firing_rate=mean_firing_rate_discretized * true_sampling_rate)

    # randomly permute data
    if randperm == True:
        data = data[torch.randperm(data.shape[0]), :]

    return data


def create_weights_binary_network(N=10, phigh=1.2, ilow=-4):
    # initialize NxN weight matrix
    Wa = np.random.uniform(high=phigh, size=(N // 2, N // 2))
    Wb = np.random.uniform(high=phigh, size=(N // 2, N // 2))
    for i in range(N // 2):
        Wa[i, i] = 0
        Wb[i, i] = 0

    # create large matrix W
    W = np.zeros(shape=(N, N))
    W[0:N // 2, 0:N // 2] = Wa
    W[N // 2:N, N // 2:N] = Wb

    # add negative connection between subgroups
    Wc = np.random.uniform(low=ilow, high=0, size=(N // 2, N // 2))
    Wd = np.random.uniform(low=ilow, high=0, size=(N // 2, N // 2))
    W[0:N // 2, N // 2:N] = Wc
    W[N // 2:N, 0:N // 2] = Wd

    return W



















