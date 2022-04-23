import numpy as np
import torch
from tqdm import tqdm


def set_to_device(rtrbm, device):
    rtrbm.errors = torch.tensor(rtrbm.errors, device=device)
    rtrbm.W = rtrbm.W.to(device).detach().clone()
    rtrbm.U = rtrbm.U.to(device).detach().clone()
    rtrbm.b_V = rtrbm.b_V.to(device).detach().clone()
    rtrbm.b_H = rtrbm.b_H.to(device).detach().clone()
    rtrbm.b_init = rtrbm.b_init.to(device).detach().clone()
    rtrbm.V = rtrbm.V.to(device).detach().clone()
    rtrbm.device = device

    return

def pairwise_moments(data1, data2):
    """Average matrix product."""
    return torch.matmul(data1, data2.T) / torch.numel(data1)


def RMSE(test, est):
    """Calculates the Root Mean Square Error of two vectors."""

    return torch.sqrt(torch.sum((test - est) ** 2) / torch.numel(test))


def nRMSE(train, test, est):
    """Calculates the normalised Root Mean Square Error of two statistics vectors, given training data statistics."""

    rmse = RMSE(test, est)

    test_shuffled = test[torch.randperm(int(test.shape[0]))]
    est_shuffled = est[torch.randperm(int(est.shape[0]))]

    rmse_shuffled = RMSE(test_shuffled, est_shuffled)
    rmse_optimal = RMSE(train, test)

    return 1 - (rmse - rmse_shuffled) / (rmse_optimal - rmse_shuffled)


def free_energy(v, W, b_V, b_H):
    """Get free energy of RBM"""
    v_term = torch.outer(v, b_V.T)
    w_x_h = torch.nn.functional.linear(v, W.T, b_H)
    h_term = torch.sum(torch.nn.functional.softplus(w_x_h))
    free_energy = torch.mean(-h_term - v_term)

    return free_energy


def get_nRMSE_moments(model, V_train, V_test, V_est, H_train, H_test, H_est, sp=0):
    """ Calculates normalised Root Mean Square Error of moments and pairwise moments """

    # <v_i>
    V_mean_train = torch.mean(V_train, 1)
    V_mean_test = torch.mean(V_test, 1)
    V_mean_est = torch.mean(V_est, 1)

    # <h_{mu}>
    H_mean_train = torch.mean(H_train, 1)
    H_mean_test = torch.mean(H_test, 1)
    H_mean_est = torch.mean(H_est, 1)

    # <v_i h_{mu}>_{model} = <v_i h_{mu}>_{model-generated data} + lamda*sign(w_{i,mu})
    VH_mgd_train = pairwise_moments(V_train, H_train)
    VH_mgd_test = pairwise_moments(V_test, H_test)
    VH_mgd_est = pairwise_moments(V_est, H_est)

    VH_mean_train = VH_mgd_train + sp * torch.sign(model.W.T)
    VH_mean_test = VH_mgd_test + sp * torch.sign(model.W.T)
    VH_mean_est = VH_mgd_est + sp * torch.sign(model.W.T)

    # <v_i v_j> - <v_i><v_j>
    VV_mean_train = pairwise_moments(V_train, V_train) - torch.outer(V_mean_train, V_mean_train)
    VV_mean_test = pairwise_moments(V_test, V_test) - torch.outer(V_mean_test, V_mean_test)
    VV_mean_est = pairwise_moments(V_est, V_est) - torch.outer(V_mean_est, V_mean_est)

    # <h_i h_j> - <h_i><h_j>
    HH_mean_train = pairwise_moments(H_train, H_train) - torch.outer(H_mean_train, H_mean_train)
    HH_mean_test = pairwise_moments(H_test, H_test) - torch.outer(H_mean_test, H_mean_test)
    HH_mean_est = pairwise_moments(H_est, H_est) - torch.outer(H_mean_est, H_mean_est)

    V_nRMSE = nRMSE(V_mean_train, V_mean_test, V_mean_est)
    H_nRMSE = nRMSE(H_mean_train, H_mean_test, H_mean_est)
    VH_nRMSE = nRMSE(VH_mean_train, VH_mean_test, VH_mean_est)
    VV_nRMSE = nRMSE(VV_mean_train, VV_mean_test, VV_mean_est)
    HH_nRMSE = nRMSE(HH_mean_train, HH_mean_test, HH_mean_est)

    return V_nRMSE, H_nRMSE, VH_nRMSE, VV_nRMSE, HH_nRMSE


def correlation(v):
    return np.corrcoef(v)


# def mutual_information(v_prob):
#    for t in range(v_prob.shape[1]-1):
#        MU[:,:,t] = torch.outer(v_prob[:,t], v_prob[:,t+1]) * torch.log()
#    return 9

def make_voxel_xyz(n, spikes, xyz, mode=1, fraction=0.5, disable_tqdm=False):
    n = n + 1  # number of voxels
    x = torch.linspace(torch.min(xyz[:, 0]), torch.max(xyz[:, 0]), n)
    y = torch.linspace(torch.min(xyz[:, 1]), torch.max(xyz[:, 1]), n)
    z = torch.linspace(torch.min(xyz[:, 2]), torch.max(xyz[:, 2]), n)

    voxel_xyz = torch.zeros((n - 1) ** 3, 3)
    voxel_spike = torch.zeros((n - 1) ** 3, spikes.shape[1])
    i = 0
    for ix in tqdm(range(n - 1), disable=disable_tqdm):
        for iy in range(n - 1):
            for iz in range(n - 1):
                condition = ((xyz[:, 0] > x[ix]) & (xyz[:, 0] < x[ix + 1]) & (xyz[:, 1] > y[iy]) & \
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

    return [voxel_spike, voxel_xyz]


def get_hidden_mean_receptive_fields(weights, coordinates, only_max_conn=False):
    """
        Computes the receptive fields of the hidden units.

        Parameters
        ----------
        VH : torch.Tensor
            The hidden layer's weight matrix.
        coordinates : torch.Tensor
            The coordinates of the visible units.
        only_max_conn : bool, optional
            If True, only the receptive field of the unit with the maximal
            connection to the hidden layer is returned.

        Returns
        -------
        torch.Tensor
            The receptive fields of the hidden units. """

    VH = weights.detach().clone()

    if only_max_conn is False: VH[VH < 0] = 0

    n_dimensions = torch.tensor(coordinates.shape).shape[0]
    N_H = VH.shape[0]

    max_hidden_connection = torch.max(VH, 0)[1]
    if n_dimensions == 1:
        rf = torch.zeros(N_H)
        for h in range(N_H):
            if only_max_conn:
                v_idx = (max_hidden_connection == h)
                rf[h] = torch.mean(coordinates[v_idx])
            else:
                rf[h] = torch.sum(VH[h, :] * coordinates / torch.sum(VH[h, :]))
    else:
        rf = torch.zeros(N_H, n_dimensions)
        for i in range(n_dimensions):
            for h in range(N_H):
                if only_max_conn:
                    v_idx = (max_hidden_connection == h)
                    rf[h, i] = torch.mean(coordinates[v_idx, i])
                else:
                    rf[h, i] = torch.sum(VH[h, :] * coordinates[:, i] / torch.sum(VH[h, :]))

    return rf


def get_param_history(parameter_history):
    """
    Returns parameter history per parameter as torch tensor
    """
    epochs = len(parameter_history)
    N_H, N_V = np.shape(parameter_history[0][0])
    W = torch.empty(epochs, N_H, N_V)
    U = torch.empty(epochs, N_H, N_H)
    b_V = torch.empty(epochs, 1, N_V)
    b_H = torch.empty(epochs, 1, N_H)
    b_init = torch.empty(epochs, 1, N_H)

    for ep, params in enumerate(parameter_history):
        W[ep] = params[0].clone().detach()
        U[ep] = params[1].clone().detach()
        b_H[ep] = params[2].clone().detach()
        b_V[ep] = params[3].clone().detach()
        b_init[ep] = params[4].clone().detach()

    return W, U, b_H, b_V, b_init


from scipy.stats.distributions import chi2


def error_ellipse(x, y, p):
    # Error ellipse with confidence interval p
    x = np.array(x)
    y = np.array(y)

    # Calculate the eigenvectors and eigenvalues
    eigenval, eigenvec = np.linalg.eig(np.cov(x, y))

    # Get the index of the largest eigenvector
    idx = np.where(eigenval == np.max(eigenval))[0]
    largest_eigenvec = eigenvec[idx, :][0]
    largest_eigenval = eigenval[idx][0]

    # Get the smallest eigenvector and eigenvalue
    idx = np.where(eigenval == np.min(eigenval))[0]
    smallest_eigenvec = eigenvec[idx, :][0]
    smallest_eigenval = eigenval[idx][0]

    # Calculate the angle between the x-axis and the largest eigenvector
    angle = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

    # This angle is between -pi and pi. shift it such that the angle is between 0 and 2pi
    if (angle < 0):
        angle = angle.ravel() + 2 * np.pi

    # Get the coordinates of the data mean
    chisquare_val = np.sqrt(chi2.ppf(p, df=2))
    theta_grid = np.linspace(0, 2 * np.pi, 100)
    a = chisquare_val * np.sqrt(largest_eigenval.ravel())
    b = chisquare_val * np.sqrt(smallest_eigenval.ravel())

    # the ellipse in x and y coordinates
    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_y_r = b * np.sin(theta_grid)

    # Define rotation matrix
    R = np.reshape(np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]),
                   [2, 2])  # change to 3,3 if you also take z

    # Rotate the ellipse to some angle phi
    r_ellipse = np.matmul(np.array([ellipse_x_r, ellipse_y_r]).T, R)

    rx = r_ellipse[:, 0] + np.mean(x)
    ry = r_ellipse[:, 1] + np.mean(y)

    # rx and ry are the coordinates of the ellipse
    return rx, ry


from scipy.stats import pearsonr


def correlation_matrix(data):
    # data.shape = [n, T]f
    population_vector = np.array(data)
    C = np.zeros((population_vector.shape[0], population_vector.shape[0]))
    for i in range(population_vector.shape[0]):
        for j in range(population_vector.shape[0]):
            C[i][j] = pearsonr(population_vector[i], population_vector[j])[0]
    return C


def cross_correlation(data, time_shift=1):
    if np.array(data.shape).shape[0]==3:

        for s in range(data.shape[2]):
            population_vector_t = np.array(data[:, time_shift:, s])
            population_vector_tm = np.array(data[:, :-time_shift, s])
            C = np.zeros([population_vector_t.shape[0], population_vector_tm.shape[0], data.shape[2]])
            for i in range(population_vector_t.shape[0]):
                for j in range(population_vector_tm.shape[0]):
                    C[i][j][s] = np.correlate(population_vector_t[i], population_vector_tm[j])
        C = np.mean(C, 2)

    elif np.array(data.shape).shape[0]==2:

        population_vector_t = np.array(data[:, time_shift:])
        population_vector_tm = np.array(data[:, :-time_shift])
        C = np.zeros([population_vector_t.shape[0], population_vector_tm.shape[0]])
        for i in range(population_vector_t.shape[0]):
            for j in range(population_vector_tm.shape[0]):
                C[i][j] = np.correlate(population_vector_t[i], population_vector_tm[j])
    return C

