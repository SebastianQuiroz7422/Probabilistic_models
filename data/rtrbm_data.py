import numpy as np
import torch
from tqdm import tqdm
from boltzmann_machines.cp_rtrbm import RTRBM
from utils.plots import plot_weights


def gen_W(N_H, N_V, mu=[1.5, 0.4], sig=[0.5, 0.5], rand_assign=0, size_chunk=[0.9, 1.1], f=0.5):

    '''
    :param N_H: Number of hidden units
    :param N_V: Number of hidden units
    :param mu: mean of populations mu[0] = mean main population, mean[!=0] = mean subpopulations
    :param sig: std of populations mu[0] = std population, std[!=0] = subpopulations
    :param rand_assign: fraction of randomly shuffeled neurons with random permutation
    :param size_chunk: randomly configure population size iff [0.9, 1.1] than  0.9*(N_V/N_H) < population size < 1.1*(N_V/N_H)
    :param f: fraction of the main populations that is negative/positive
    :return: Weight matrix of hiddens - visibles
    '''
    # initialze fully connected random weight matrix
    W = torch.zeros((N_H, N_V), dtype=torch.float)
    mu = torch.tensor(mu)
    sig = torch.tensor(sig)

    # Assign random sized populations to each N_H (how many connections has each N_H at connectivity=minimum connectivity)
    chunks = N_V * torch.ones(N_H - 1)
    while torch.sum(chunks) > N_V:
        chunks = torch.randint(low=int(size_chunk[0] * (N_V / N_H)), high=int(size_chunk[1] * (N_V / N_H)),
                               size=(N_H - 1,))
    chunks = torch.cat([torch.tensor([0]), torch.cat([chunks, torch.tensor([N_V - torch.sum(chunks)])])])
    chunks = [torch.sum(chunks[:h + 1]) for h in range(N_H + 1)]

    # Randomly assign N_V idx to each HU population (randomly can be changed with rand_assign)
    temp = torch.arange(N_V)
    rand_idx = torch.randperm(N_V)[:int(N_V * rand_assign)]
    temp[rand_idx] = rand_idx[torch.randperm(rand_idx.shape[0])]
    randperm = [temp[chunks[h]:chunks[h + 1]] for h in range(N_H)]

    # Define main populations
    idxh = torch.randperm(N_H)[:int(N_H * f)]

    for h in range(N_H):
        if torch.sum(h == idxh) == 1:
            W[h, randperm[h]] = mu[0] + sig[0] * torch.randn(size=randperm[h].shape, dtype=torch.float)
        else:
            W[h, randperm[h]] = -mu[0] + sig[0] * torch.randn(size=randperm[h].shape, dtype=torch.float)

    # shuffle
    mu = mu[1:]
    sig = sig[1:]
    idxh = torch.randperm(len(mu))
    mu = mu[idxh]
    sig = sig[idxh]

    idxh = torch.randperm(N_H)
    W = W[idxh, :]

    # Define sub-populations
    n_pop = len(mu)
    size_pop = torch.randint(low=1, high=N_H, size=(n_pop,))
    sub_con = torch.ones((N_H, N_H)) - torch.eye(N_H)[idxh, :]
    start = [0, 0]

    for n in range(n_pop):
        start = torch.randint(low=0, high=N_H, size=(1, 2,))[0]
        while start[0] == start[1] or sub_con[start[0], start[1]] == 0:
            start = torch.randint(low=0, high=N_H, size=(1, 2,))[0]
        sub_con[start[0], start[1]] = n + 1
        tempm = start.detach().clone()

        for i in range(size_pop[n]):
            # random select step -1, 0, 1
            step = torch.randint(low=-1, high=2, size=(1, 2,))[0]

            # make sure its not diagonal
            while torch.abs(step[0]) == torch.abs(step[1]):
                step = torch.randint(low=-1, high=2, size=(1, 2,))[0]

            temp = tempm + step
            j = 0

            # make sure its not out of bounds and make sure its not already occupied
            while temp[0] < 0 or temp[0] > N_H - 1 or temp[1] < 0 or temp[1] > N_H - 1 or \
                    j > 100 or sub_con[temp[0], temp[1]] == 0:
                # random select step -1, 0, 1
                step = torch.randint(low=-1, high=2, size=(1, 2,))[0]

                # make sure step is not diagonal
                while torch.abs(step[0]) == torch.abs(step[1]):
                    step = torch.randint(low=-1, high=2, size=(1, 2,))[0]
                temp = tempm + step
                j += 1

            if j < 100:
                tempm = temp.detach().clone()
                sub_con[temp[0], temp[1]] = n + 1

    # take only the sub populations
    sub_con[sub_con <= 1] = 0
    sub_conn = torch.zeros_like(W)

    # set to the right shape
    for h in idxh:
        sub_conn[:, randperm[h]] = sub_con[:, h].repeat(randperm[h].shape[0]).reshape([randperm[h].shape[0], N_H]).T

    # add to W
    for p in range(1, n_pop):
        W += (mu[p] + sig[p] * torch.randn(size=W.shape, dtype=torch.float)) * (sub_conn == p + 1)

    return W


def gen_U(N_H, nabla=2, connectivity=0, std_max=40):
    '''
    :param N_H: Number of hidden units
    :param nabla: Norm of U, for a more dynamically system use nabla = torch.sum(W,1) (norm of W over visibles)
    :param connectivity: fraction of connectivity in [0, 1], 0 means only 1 hidden is connected to another hidden
                         and 1 means a fully connected system
    :param std: Minimal std of norm of U till acceptance
    :return: Weight matrix of hiddens(t) - hiddens(t-1)

    '''
    std = 500
    it = 0
    while std > std_max or it < 100:
        ## initialize parameters
        sparcity = 1 - connectivity
        sp = 1 - 1 / N_H

        U = torch.eye(N_H) * torch.randn((N_H, N_H), dtype=torch.float)
        U = U[torch.randperm(N_H), :]
        # compute the number of zeros we need to have in the weight matrix to obtain the predefined connectivity
        sp = torch.sum(U == 0) / U.numel()
        conn = 1 - sp
        n_zeros = int(torch.sum(U == 0) - sparcity * U.numel())

        if n_zeros > 0:
            # randomly add values to the weight matrix in order to obtain the predefined connectivity
            idx = torch.where(U.ravel() == 0)[0]
            idx = idx[torch.randperm(idx.shape[0])[:n_zeros]]
            U.ravel()[idx] = torch.randn(n_zeros, dtype=torch.float)
        else:
            print('Minimum connectivity is: ' + str(conn))

        for i in range(N_H):
            idx = U[i, :] != 0
            if torch.sum(idx) > 1:
                U[i, idx] = (U[i, idx] - torch.mean(U[i, idx])) / torch.std(U[i, idx])

            temp = torch.randn(N_H) * (U[i, :] != 0)
            if isinstance(nabla, list) or isinstance(nabla, torch.Tensor):
                if len(nabla) != N_H:
                    raise ValueError('nabla should be an int or a list/torch.Tensor with length N_H')
                temp = nabla[i] * temp / torch.sum(temp)
            else:
                temp = nabla * temp / torch.sum(temp)
            U[i, :] += temp

        std = torch.std(U.ravel())
        it += 1

    if it == 100:
        raise ValueError('Try again with different W, change std, norm U, or increase connectivity')
    return U



def get_rtrbm_data(N_H=5, N_V=100,# system size
                   T=30, n_batches=300, device='cpu', pre_gibbs_k=100, gibbs_k=100, mode=1, # infer data
                   mu=None, sig=None, rand_assign=None, size_chunk=None, f=None, # Generate W matrix
                   nabla=None, connectivity=None, std_max=None, # Generate U matrix
                   show_figure=True):


    if mu is None:
        mu = [1.2, 0.2, 0.2, 0.1, 0.05, -0.05, -0.1, -0.2, -0.2]
    if sig is None:
        sig = [0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2]
    if len(mu) != len(sig):
        raise ValueError('mu and sig should be the same length')
    if rand_assign is None:
        rand_assign = 0
    if size_chunk is None:
        size_chunk = [0.85, 1.15]
    if f is None:
        f=0.5

    # Generate W matrix
    W = gen_W(N_H, N_V, mu=mu, sig=sig, rand_assign=rand_assign, size_chunk=size_chunk, f=f)

    if nabla is None:
        nabla = -torch.sum(W, 1) - (torch.randn(N_H) + 2 * torch.sign(torch.sum(W, 1)))
    if connectivity is None:
        connectivity=1
    if std_max is None:
        std_max = N_V / N_H

    # Generate U matrix
    U = gen_U(N_H, nabla=nabla, connectivity=connectivity, std_max=std_max)


    # Plot heatmap of both matrices
    if show_figure:
        plot_weights(W, U, figsize=(10, 4))


    # Initialize RTRBM and data in batches
    data = torch.zeros(N_V, T, device=device, dtype=torch.float)
    rtrbm = RTRBM(data, N_H=N_H, device=device, no_bias=False, debug_mode=False)

    data = torch.zeros(N_V, T, n_batches, dtype=torch.float)
    rt = torch.zeros(N_H, T, n_batches)
    rtrbm.W = torch.tensor(W, device=device, dtype=torch.float)
    rtrbm.U = torch.tensor(U, device=device, dtype=torch.float)

    # sample data from RTRBM
    for batch in tqdm(range(n_batches)):
        v_start = (torch.rand(N_V) > 0.2) * 1.0
        data[..., batch], rt[..., batch] = rtrbm.sample(v_start.type(torch.float).to(device),
                                                          chain=T,
                                                          pre_gibbs_k=pre_gibbs_k,
                                                          gibbs_k=gibbs_k,
                                                          mode=mode,
                                                          disable_tqdm=True)


    return rtrbm, data, rt



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    rtrbm, data, rt = get_rtrbm_data(N_H=5,
                   N_V=100,
                   T=30,
                   n_batches=5,
                   device='cpu',
                   pre_gibbs_k=100,
                   gibbs_k=100,
                   mode=1,
                   show_figure=True)
    # torch.save(rtrbm, open(r'C:\Users\sebas\OneDrive\Intern\rtrbm_master\results\Reproduce_rtrbm\rtrbm_paper0', 'wb'))
    print(torch.mean(data))
    sns.heatmap(data[:, :, 0].cpu())
    plt.show()
