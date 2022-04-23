import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import seaborn as sns
from tqdm import tqdm
import random
import matplotlib.animation as animation
import matplotlib.colors as colors
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from matplotlib import cm
from utils.funcs import get_param_history


def hist_strongest_weights(weights, threshold, ax=None):
    if ax is None:
        ax = plt.subplot()

    connectivity = torch.sum(weights > threshold, 0)
    n, counts = torch.unique(connectivity, return_counts=True)
    ax.bar(n, counts / torch.sum(counts))
    ax.set_xlabel('# Strong weights per neuron', fontsize=15)
    ax.set_ylabel('PDF', fontsize=15)
    return ax


def plot_mean_std_param_history(parameter_history):
    """
    Plots the mean and standard deviation of every parameter over epochs
    """
    W, U, b_H, b_V, b_init = get_param_history(parameter_history)
    epochs = np.arange(W.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    params = [W, U, b_H, b_V, b_init]
    for param in params:
        axes[0].plot(epochs, torch.mean(param, (1, 2)))
        axes[1].plot(epochs, torch.std(param, (1, 2)))
    axes[0].legend(['W', 'U', 'b_H', 'b_V', 'b_init'])
    axes[0].set_ylabel('Means', fontsize=15)
    axes[1].set_ylabel('STDs', fontsize=15)
    axes[1].set_xlabel('epochs', fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_weights(W, U, figsize=(10, 4), fs = 12):

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(W, ax=axes[0])
    axes[0].set_ylabel('$h^{[t]}$', color='#2F5597', fontsize=fs, rotation=0, labelpad=25)
    axes[0].set_xlabel('$v^{[t]}$', color='#C55A11', fontsize=fs)
    axes[0].set_title('$W$', fontsize=1.25 * fs)
    # axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].xaxis.set_ticks([])
    axes[0].yaxis.set_ticks([])

    sns.heatmap(U, ax=axes[1])
    # axes[1].set_xlabel('$h^{[t]}$', color='#2F5597', fontsize=fs)
    axes[1].set_xlabel('$h^{[t-1]}$', color='#2F5597', fontsize=fs)
    axes[1].set_title('$U$', fontsize=1.25 * fs)
    # axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].xaxis.set_ticks([])
    axes[1].yaxis.set_ticks([])
    plt.tight_layout()

    return plt.gca()


def raster_plot(data, xticklabels=100, figsize=(15, 4), title='Spiking pattern'):
    plt.figure(figsize=figsize)
    colors = ['white', 'black']
    cmap = LinearSegmentedColormap.from_list('', colors, 2)
    sns.heatmap(data, cbar=False, cmap=cmap, vmin=0, vmax=1, xticklabels=xticklabels)
    plt.title(title, fontsize=15)
    plt.xlabel('time', fontsize=12)
    plt.ylabel('# neuron', fontsize=12)
    return plt.gca()


def plot_reconstruction_error(errors, figsize=(5, 4), axes=None, fs=12, title=None):
    if axes is None:
        plt.figure(figsize=figsize)
        plt.plot(errors)
        if title is None:
            plt.title('Reconstruction error', fontsize=1.25 * fs)
        else:
            plt.title(title, fontsize=1.25 * fs)
        plt.xlabel('epochs', fontsize=fs)
        plt.ylabel('normalised reconstruction error', fontsize=fs)
    else:
        axes.plot(errors)
        if title is None:
            axes.set_title('Reconstruction error', fontsize=1.25 * fs)
        else:
            axes.set_title(title, fontsize=1.25 * fs)
        axes.set_xlabel('epochs', fontsize=fs)
        axes.set_ylabel('normalised reconstruction error', fontsize=fs)

    return plt.gca()


def plot_rtrbm_reestimate_weights(rtrbm_original, rtrbm_estimated, figsize=(10, 4), fs=12):
    # get weights
    W_original = rtrbm_original.W.detach().clone()
    U_original = rtrbm_original.U.detach().clone()
    W_estimated = rtrbm_estimated.W.detach().clone()
    U_estimated = rtrbm_estimated.U.detach().clone()
    N_H = U_original.shape[0]

    # calculate correlation and reshuffle weights
    corr = np.zeros((N_H, N_H))
    shuffle_idx = np.zeros((N_H))
    for i in range(N_H):
        for j in range(N_H):
            corr[i, j] = np.correlate(W_original[i, :], W_estimated[j, :])
        shuffle_idx[i] = np.argmax(corr[i, :])

    W_estimated[:, :] = W_estimated[shuffle_idx, :]
    U_estimated[:, :] = U_estimated[shuffle_idx, :]
    U_estimated[:, :] = U_estimated[:, shuffle_idx]

    r2_W = np.corrcoef(W_estimated.ravel(), W_original.ravel())[0, 1] ** 2
    r2_U = np.corrcoef(U_estimated.ravel(), U_original.ravel())[0, 1] ** 2

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(W_original.ravel(), W_estimated.ravel(), 'o')
    mini = min(W_original.ravel().min(), W_estimated.ravel().min())
    maxi = max(W_original.ravel().min(), W_estimated.ravel().min())
    maxi = int(max(abs(mini), abs(maxi)))
    axes[0].text(0, -0.6 * maxi, r'$R^2 = %.2f$' % r2_W, fontsize=fs)
    axes[0].plot([-maxi, maxi], [-maxi, maxi], ':')
    axes[0].set_xticks([-maxi, 0, maxi])
    axes[0].set_yticks([-maxi, 0, maxi])
    axes[0].set_xlabel('Original weights', fontsize=fs)
    axes[0].set_ylabel('Estimated weights', fontsize=fs)
    axes[0].set_title('Visible to hidden weights $W$', fontsize=1.25 * fs)


    axes[1].plot(U_original.ravel(), U_estimated.ravel(), 'o')
    mini = min(U_original.ravel().min(), U_estimated.ravel().min())
    maxi = max(U_original.ravel().min(), U_estimated.ravel().min())
    maxi = int(max(abs(mini), abs(maxi)))
    axes[1].text(0, -0.6 * maxi, r'$R^2 = %.2f$' % r2_U, fontsize=fs)
    axes[1].plot([-maxi, maxi], [-maxi, maxi], ':', 'grey')
    axes[1].set_xticks([-maxi, 0, maxi])
    axes[1].set_yticks([-maxi, 0, maxi])
    axes[1].set_xlabel('Original weights', fontsize=fs)
    axes[1].set_ylabel('Estimated weights', fontsize=fs)
    axes[1].set_title('Hidden to hidden weights $U$', fontsize=1.25 * fs)

    plt.show()
    return


def plot_different_infer(model, data, t_start_infer=8, t_extra=0):
    
    fig, ax = plt.subplots(6, figsize=(20,20))
    
    sns.heatmap(data, ax=ax[0], cbar=False)
    ax[0].set_title('True data')
    
    V, _ = model.infer(data[:,0:t_start_infer], t_extra=t_extra, pre_gibbs_k=0, gibbs_k=1, disable_tqdm=True)
    sns.heatmap(V, ax=ax[1], cbar=False)
    ax[1].set_title('1 Gibbs sample')
    
    V, _ = model.infer(data[:,0:t_start_infer], t_extra=t_extra, pre_gibbs_k=0, gibbs_k=10, mode=2, disable_tqdm=True)
    sns.heatmap(V, ax=ax[2], cbar=False)
    ax[2].set_title('mean of 10 Gibbs samples')
    
    V, _ = model.infer(data[:,0:t_start_infer], t_extra=t_extra, pre_gibbs_k=0, gibbs_k=100, mode=2, disable_tqdm=True)
    sns.heatmap(V, ax=ax[3], cbar=False)
    ax[3].set_title('mean of 100 Gibbs samples')
       
    V, _ = model.infer(data[:,0:t_start_infer], t_extra=t_extra, pre_gibbs_k=0, gibbs_k=100, mode=1, disable_tqdm=True)
    sns.heatmap(V, ax=ax[4], cbar=False)
    ax[4].set_title('last of 100 Gibbs samples')
    
    V, _ = model.infer(data[:,0:t_start_infer], t_extra=t_extra, pre_gibbs_k=0, gibbs_k=100, mode=3, disable_tqdm=True)
    sns.heatmap(V, ax=ax[5], cbar=False)
    ax[5].set_title('most probable of 100 Gibbs samples')
        
    plt.show()

    
    for x in ax[0:5]:
        x.set_xticks([])
    
    return 


def plot_weights_true_reconstructed(VH_true, HH_true, VH_recon, HH_recon, normalize_weights=False):
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    if normalize_weights:
        VH_true /= torch.max(torch.abs(VH_true))
        HH_true /= torch.max(torch.abs(HH_true))
        VH_recon /= torch.max(torch.abs(VH_recon))
        HH_recon /= torch.max(torch.abs(HH_recon))

    sns.heatmap(VH_true.T, ax=ax[0,0], cbar=True, cbar_ax=cbar_ax, vmin=-1, vmax=1)
    ax[0,0].set_xticklabels([])
    ax[0,0].set_ylabel("visible neurons")
    ax[0,0].set_title("true")
    sns.heatmap(HH_true, ax=ax[1,0], cbar=False, vmin=-1, vmax=1)
    ax[1,0].set_ylabel("hidden neurons")
    ax[1,0].set_xlabel("hidden neurons")
    sns.heatmap(VH_recon.T, ax=ax[0,1], cbar=False, vmin=-1, vmax=1)
    ax[0,1].set_title("reconstructed")
    #ax[0,1].set_xticklabels([])
    #ax[0,1].set_yticklabels([])
    sns.heatmap(HH_recon, ax=ax[1,1], cbar=False, vmin=-1, vmax=1)
    #ax[1,1].set_yticklabels([])
    ax[1,1].set_xlabel("hidden neurons")
    #fig.tight_layout()
    plt.show()

    return

def plot_visible_hidden_weights_activations(VH, HH, VT, HT):
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    VH /= torch.max(torch.abs(VH))
    HH /= torch.max(torch.abs(HH))
    
    sns.heatmap(VH.T, ax=ax[0,0], cbar=True, cbar_ax=cbar_ax, vmin=-1, vmax=1)
    ax[0,0].set_xticklabels([])
    ax[0,0].set_ylabel("visible neurons")
    ax[0,0].set_title("nomalized weights")
    sns.heatmap(HH, ax=ax[1,0], cbar=False, vmin=-1, vmax=1)
    ax[1,0].set_ylabel("hidden neurons")
    ax[1,0].set_xlabel("hidden neurons")
    sns.heatmap(VT, ax=ax[0,1], cbar=False, vmin=-1, vmax=1)
    ax[0,1].set_title("activations")
    ax[0,1].set_xticklabels([])
    ax[0,1].set_yticklabels([])
    sns.heatmap(HT, ax=ax[1,1], cbar=False, vmin=-1, vmax=1)
    ax[1,1].set_yticklabels([])
    ax[1,1].set_xlabel("time")
    fig.tight_layout(rect=[0, 0, .9, 1])
    plt.show()

    return


def plot_true_sampled(V_data, H_data, V_sampled, H_sampled):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparison True and Sampled data')

    axes[0, 0].set_title('True')
    sns.heatmap(V_data, ax=axes[0, 0], cbar=False)

    axes[0, 1].set_title('Sampled')
    sns.heatmap(V_sampled, ax=axes[0, 1], cbar=False)

    axes[1, 0].set_title('True Hiddens')
    sns.heatmap(H_data, ax=axes[1, 0], cbar=False)

    axes[1, 1].set_title('Sampled Hiddens')
    sns.heatmap(H_sampled, ax=axes[1, 1], cbar=False)

    return


def plot_true_sampled_hiddens_stimulus(data, sampled, hiddens, stimulus=None, train_test_line=0):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparison True and Sampled data')

    axes[0, 0].set_title('Sampled')
    sns.heatmap(sampled, ax=axes[0, 0], cbar=False)
    if train_test_line != 0:
        axes[0, 0].vlines(train_test_line, 0, data.shape[0])

    axes[0, 1].set_title('True')
    if train_test_line != 0:
        axes[0, 1].vlines(train_test_line, 0, data.shape[0])
    sns.heatmap(data, ax=axes[0, 1], cbar=False)

    axes[1, 0].set_title('Hidden nodes')
    if train_test_line != 0:
        axes[1, 0].vlines(train_test_line, 0, data.shape[0])
    sns.heatmap(hiddens, ax=axes[1, 0], cbar=False)

    if stimulus is not None:
        axes[1, 1].set_title('Stimulus')
        axes[1, 1].plot(stimulus)

    return


def pairwise_moments(data1, data2):
    return torch.matmul(data1, data2.T) / data1.shape[1]


def moments(V_data, H_data, V_samples, H_samples):
    means_V_data = torch.mean(V_data, 1)
    means_V_samples = torch.mean(V_samples, 1)
    means_H_data = torch.mean(H_data, 1)
    means_H_samples = torch.mean(H_samples, 1)

    ind_VV = np.triu_indices(n=V_data.shape[0], k=1)
    ind_HH = np.triu_indices(n=H_data.shape[0], k=1)
    ind_VH = np.triu_indices(n=V_data.shape[0], m=H_data.shape[0], k=1)

    pw_VV_data = pairwise_moments(V_data, V_data)[ind_VV]
    pw_VV_samples = pairwise_moments(V_samples, V_samples)[ind_VV]
    pw_VH_data = pairwise_moments(V_data, H_data)[ind_VH]
    pw_VH_samples = pairwise_moments(V_samples, H_samples)[ind_VH]
    pw_HH_data = pairwise_moments(H_data, H_data)[ind_HH]
    pw_HH_samples = pairwise_moments(H_samples, H_samples)[ind_HH]

    _, _, r_V, _, _ = linregress(means_V_data, means_V_samples)
    _, _, r_H, _, _ = linregress(means_H_data, means_H_samples)
    _, _, r_VV, _, _ = linregress(pw_VV_data, pw_VV_samples)
    _, _, r_VH, _, _ = linregress(pw_VH_data, pw_VH_samples)
    _, _, r_HH, _, _ = linregress(pw_HH_data, pw_HH_samples)

    return means_V_data, means_V_samples, means_H_data, means_H_samples, pw_VV_data, \
           pw_VV_samples, pw_HH_data, pw_HH_samples, pw_VH_data, pw_VH_samples, \
           r_V, r_H, r_VV, r_VH, r_HH


def plot_moments(V_data, H_data, V_samples, H_samples):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    means_V_data, means_V_samples, means_H_data, means_H_samples, pw_VV_data, \
    pw_VV_samples, pw_HH_data, pw_HH_samples, pw_VH_data, pw_VH_samples, \
    r_V, r_H, r_VV, r_VH, r_HH = moments(V_data, H_data, V_samples, H_samples)

    ax = axes[0, 0]
    ax.plot(means_V_data, means_V_samples, 'bo')
    ax.plot([-1, 1], [-1, 1], ':', color='k')
    ax.set_xlim([torch.min(means_V_data) - 0.1, torch.max(means_V_data) + 0.1])
    ax.set_ylim([torch.min(means_V_samples) - 0.1, torch.max(means_V_samples) + 0.1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Training data")
    ax.set_ylabel("Sampled data")
    ax.set_title("Means V-Units, r-value: {}".format(r_V))

    ax = axes[0, 1]
    ax.plot(means_H_data, means_H_samples, 'o', color='orange')
    ax.plot([-1, 1], [-1, 1], ':', color='k')
    ax.set_xlim([torch.min(means_H_data) - 0.1, torch.max(means_H_data) + 0.1])
    ax.set_ylim([torch.min(means_H_samples) - 0.1, torch.max(means_H_samples) + 0.1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Training data")
    ax.set_ylabel("Sampled data")
    ax.set_title("Means H-Units, r-value: {}".format(r_H))

    ax = axes[0, 2]
    ax.plot(pw_VH_data, pw_VH_samples, 'go')
    ax.plot([-1, 1], [-1, 1], ':', color='k')
    ax.set_xlim([torch.min(pw_VH_data) - 0.1, torch.max(pw_VH_data) + 0.1])
    ax.set_ylim([torch.min(pw_VH_samples) - 0.1, torch.max(pw_VH_samples) + 0.1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Training data")
    ax.set_ylabel("Sampled data")
    ax.set_title("Pairwise moments V/H units, r-value: {}".format(r_VH))

    ax = axes[1, 0]
    ax.plot(pw_VV_data, pw_VV_samples, 'ro')
    ax.plot([-1, 1], [-1, 1], ':', color='k')
    ax.set_xlim([torch.min(pw_VV_data) - 0.1, torch.max(pw_VV_data) + 0.1])
    ax.set_ylim([torch.min(pw_VV_samples) - 0.1, torch.max(pw_VV_samples) + 0.1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Training data")
    ax.set_ylabel("Sampled data")
    ax.set_title("Pairwise moments V/V units, r-value: {}".format(r_VV))

    ax = axes[1, 1]
    ax.plot(pw_HH_data, pw_HH_samples, 'o', color='purple')
    ax.plot([-1, 1], [-1, 1], ':', color='k')
    ax.set_xlim([torch.min(pw_HH_data) - 0.1, torch.max(pw_HH_data) + 0.1])
    ax.set_ylim([torch.min(pw_HH_samples) - 0.1, torch.max(pw_HH_samples) + 0.1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Training data")
    ax.set_ylabel("Sampled data")
    ax.set_title("Pairwise moments H/H units, r-value: {}".format(r_HH))

    fig.tight_layout()
    plt.show()


def plot_weights_log_distribution(weights, ax=None):
    if ax is None:
        ax = plt.subplot()
    sns.kdeplot(weights.flatten(), log_scale=[0, 10], ax=ax)
    #fig.set_axis_labels('Weight value', 'PDF')
    return


def plot_weights_log_distribution_compare(weights1, weights2, label1='label1', label2='label2', ymin=1e-3, ymax=1e2):
    fig, ax = plt.subplots()
    sns.kdeplot(weights1.flatten(), legend=False, log_scale=[0, 10], ax=ax)
    sns.kdeplot(weights2.flatten(), legend=False, log_scale=[0, 10], ax=ax)
    ax.legend(loc='upper left', labels=[label1, label2])
    ax.set_ylim([ymin, ymax])
    return


def plot_pw_correlations(V_data, V_samples, H_data, H_samples, figsize=(10, 15)):
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    pw_VV_data = pairwise_moments(V_data, V_data)
    pw_VV_samples = pairwise_moments(V_samples, V_samples)
    pw_VH_data = pairwise_moments(V_data, H_data)
    pw_VH_samples = pairwise_moments(V_samples, H_samples)
    pw_HH_data = pairwise_moments(H_data, H_data)
    pw_HH_samples = pairwise_moments(H_samples, H_samples)

    ax = axes[0, 0]
    sns.heatmap(pw_VH_data, ax=ax)
    ax.set_title('Pairwise moments V/H data')

    ax = axes[0, 1]
    sns.heatmap(pw_VH_samples, ax=ax)
    ax.set_title('Pairwise moments V/H samples')

    ax = axes[1, 0]
    sns.heatmap(pw_VV_data, ax=ax)
    ax.set_title('Pairwise moments V/V data')

    ax = axes[1, 1]
    sns.heatmap(pw_VV_data, ax=ax)
    ax.set_title('Pairwise moments V/V samples')

    ax = axes[2, 0]
    sns.heatmap(pw_HH_data, ax=ax)
    ax.set_title('Pairwise moments H/H data')

    ax = axes[2, 1]
    sns.heatmap(pw_HH_data, ax=ax)
    ax.set_title('Pairwise moments H/H samples')

    fig.tight_layout()
    plt.show()

def plot_effective_coupling_VH_HH(VH, HH, v, rt):
    # W.shape = [H, V]

    # variance matrix
    var_h_matrix = torch.reshape(torch.var(rt, 1).repeat(VH.shape[1]), [VH.shape[1], VH.shape[0]]).T
    var_v_matrix = torch.reshape(torch.var(v, 1).repeat(VH.shape[0]), [VH.shape[0], VH.shape[1]])

    # effective coupling VH
    Je_VH = torch.mm(VH.T, VH * var_h_matrix)/VH.shape[1]**2
    Je_HV = torch.mm(VH * var_v_matrix, VH.T)/VH.shape[0]**2

    # effective coupling HH h = H[t-1] and reshape var_h_matrix
    var_h_matrix = torch.reshape(torch.var(rt, 1).repeat(HH.shape[1]), [HH.shape[1], HH.shape[0]]).T
    Je_Hh = torch.mm(HH.T, HH * var_h_matrix)/VH.shape[0]**2
    Je_hH = torch.mm(HH, HH.T * var_h_matrix)/VH.shape[0]**2

    fig, axes = plt.subplots(3,2,figsize=(16,16))
    sns.heatmap(VH , ax = axes[0,0])
    axes[0,0].set_ylabel("Hidden nodes", fontsize=18)
    axes[0,0].set_xlabel("Visible nodes", fontsize=18)
    axes[0,0].set_title('VH', fontsize=18)
    axes[0,0].tick_params(axis='both', which='major', labelsize=10)


    sns.heatmap(Je_VH , ax = axes[1,0])
    axes[1,0].set_ylabel("Visibel nodes", fontsize=18)
    axes[1,0].set_xlabel("Visible nodes", fontsize=18)
    axes[1,0].set_title('Effective coupling V', fontsize=18)
    axes[1,0].tick_params(axis='both', which='major', labelsize=10)

    sns.heatmap(Je_HV , ax = axes[2,0])
    axes[2,0].set_ylabel("Hidden nodes", fontsize=18)
    axes[2,0].set_xlabel("Hidden nodes", fontsize=18)
    axes[2,0].set_title('Effective coupling H', fontsize=18)
    axes[2,0].tick_params(axis='both', which='major', labelsize=10)

    sns.heatmap(HH , ax = axes[0,1])
    axes[0,1].set_ylabel("Hidden nodes[t-1]", fontsize=18)
    axes[0,1].set_xlabel("Hidden nodes[t]", fontsize=18)
    axes[0,1].set_title('HH', fontsize=18)
    axes[0,1].tick_params(axis='both', which='major', labelsize=10)

    sns.heatmap(Je_Hh , ax = axes[1,1])
    axes[1,1].set_ylabel("Hidden nodes [t]", fontsize=18)
    axes[1,1].set_xlabel("Hidden nodes [t]", fontsize=18)
    axes[1,1].set_title('Effective coupling H[t]', fontsize=18)
    axes[1,1].tick_params(axis='both', which='major', labelsize=10)

    sns.heatmap(Je_hH , ax = axes[2,1])
    axes[2,1].set_ylabel("Hidden nodes[t-1]", fontsize=18)
    axes[2,1].set_xlabel("Hidden nodes[t-1]", fontsize=18)
    axes[2,1].set_title('Effective coupling H[t-1]', fontsize=18)
    axes[2,1].tick_params(axis='both', which='major', labelsize=10)

    plt.show()
    return

def plot_effective_coupling_VH(VH, v, h):
    # W.shape = [H, V]

    # variance matrix
    var_h_matrix = torch.reshape(torch.var(h, 1).repeat(VH.shape[1]), [VH.shape[1], VH.shape[0]]).T
    var_v_matrix = torch.reshape(torch.var(v, 1).repeat(VH.shape[0]), [VH.shape[0], VH.shape[1]])

    # effective coupling VH
    Je_VH = torch.mm(VH.T, VH * var_h_matrix)/VH.shape[1]**2
    Je_HV = torch.mm(VH * var_v_matrix, VH.T)/VH.shape[0]**2

    fig, axes = plt.subplots(3,1,figsize=(8,16))
    sns.heatmap(VH , ax = axes[0])
    axes[0].set_ylabel("Hidden nodes", fontsize=28)
    axes[0].set_xlabel("Visible nodes", fontsize=28)
    axes[0].set_title('VH', fontsize=28)
    axes[0].tick_params(axis='both', which='major', labelsize=20)

    sns.heatmap(Je_VH , ax = axes[1])
    axes[1].set_ylabel("Visibel nodes", fontsize=28)
    axes[1].set_xlabel("Visible nodes", fontsize=28)
    axes[1].set_title('Effective coupling V', fontsize=28)
    axes[1].tick_params(axis='both', which='major', labelsize=20)

    sns.heatmap(Je_HV , ax = axes[2])
    axes[2].set_ylabel("Hidden nodes")
    axes[2].set_xlabel("Hidden nodes")
    axes[2].set_title('Effective coupling H')
    axes[2].tick_params(axis='both', which='major', labelsize=20)

    plt.show()
    return


def plot_weights_heatmap_sum(weights, print_sparsity=False, th=1e-3):
    V, H = np.shape(weights)

    g = sns.JointGrid()
    g.ax_marg_y.cla()
    g.ax_marg_x.cla()
    sns.heatmap(weights, ax=g.ax_joint, cbar=False)
    g.ax_marg_y.barh(np.arange(0.5, V), torch.sum(weights, 1))
    g.ax_marg_x.bar(np.arange(0.5, H), torch.sum(weights, 0))

    g.ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
    g.ax_marg_y.tick_params(axis='y', left=False, labelleft=False)
    if print_sparsity:
        print('Sparcity:\t',
              float(((weights.ravel() < th) & (weights.ravel() > -th)).sum() / (weights.shape[0] * weights.shape[1])))


def plot_weights_heatmap_sum_abs(weights, print_sparsity=False, th=1e-3):
    V, H = np.shape(weights)

    g = sns.JointGrid()
    g.ax_marg_y.cla()
    g.ax_marg_x.cla()
    sns.heatmap(weights, ax=g.ax_joint, cbar=False)
    g.ax_marg_y.barh(np.arange(0.5, V), torch.sum(torch.abs(weights), 1))
    g.ax_marg_x.bar(np.arange(0.5, H), torch.sum(torch.abs(weights), 0))

    g.ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
    g.ax_marg_y.tick_params(axis='y', left=False, labelleft=False)

    if print_sparsity:
        print('Sparcity:\t',
              float(((weights.ravel() < th) & (weights.ravel() > -th)).sum() / (weights.shape[0] * weights.shape[1])))


def animation_param_per_epoch(file_dir, param='W'):

    rtrbm = torch.load(open(file_dir, 'rb'), map_location='cpu')

    if param == 'W':
        X = rtrbm.parameter_history[0]
    if param == 'U':
        X = rtrbm.parameter_history[1]
    if param == 'b_H':
        X = rtrbm.parameter_history[2]
    if param == 'b_V':
        X = rtrbm.parameter_history[3]
    if param == 'b_init':
        X = rtrbm.parameter_history[4]

    if param == 'W' or 'U':
        n_epochs = X.shape[2]
        snapshots_X = [X[:, :, i] for i in range(n_epochs)]

        # plt.clf()
        fig, axes = plt.subplots(2, 1, figsize=(6, 9))
        im_X = axes[0].imshow(snapshots_X[0], interpolation='none', aspect='auto',
                              vmin=X.ravel().min(), vmax=X.ravel().max())

    else:
        n_epochs = X.shape[1]
        snapshots_X = [X[:, i] for i in range(n_epochs)]

        # plt.clf()
        fig, axes = plt.subplots(2, 1, figsize=(6, 9))
        im_X = axes[0].bar(np.arange(snapshots_X[0].shape[0]), snapshots_X[0], interpolation='none', aspect='auto',
                              vmin=X.ravel().min(), vmax=X.ravel().max())


    # add another axes at the top left corner of the figure
    axtext = fig.add_axes([0.0, 0.95, 0.1, 0.05])
    # turn the axis labels/spines/ticks off
    axtext.axis("off")

    time = axtext.text(0.5, 0.5, str(0), ha="left", va="top")

    axes[0].set_xlabel('Hiddens')
    axes[0].set_ylabel('Visibles')
    axes[0].set_title(param)

    def animate_func(i):
        im_X.set_array(snapshots_X[i])
        time.set_text(str(i))
        return [im_X, time]

    ani = animation.FuncAnimation(
        fig,
        animate_func,
        frames=n_epochs,
        interval=1,  # in ms
        blit=False)

    plt.show()

def plot_pca(X):
    from sklearn.decomposition import PCA
    # X is the data set, could be the weight matrix
    pca = PCA()
    pca.fit(X)
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    ax1.set_xlabel("PC", fontsize=15)
    ax1.set_ylabel("EVR", fontsize=15)

    ax2 = plt.subplot(1, 2, 2)
    cumulative_EVR = [sum(pca.explained_variance_ratio_[:i]) for i in range(len(pca.explained_variance_ratio_))]
    ax2.plot(range(len(pca.explained_variance_ratio_)), cumulative_EVR)
    ax2.set_xlabel("# of retained PCs", fontsize=15)
    ax2.set_ylabel("cumulative EVR", fontsize=15)
    plt.show()


def plot_first3PCA(X, labels='None'):
    from sklearn.decomposition import PCA
    if labels == 'None':
        labels = np.arange(len(X))

    pca = PCA(n_components=3)  # this PCA model will retain only the first 3 components
    pca_results = pca.fit_transform(X)  # fit_transform() finds the n_components PC and projects the data along them
    x = pca_results.T[0]
    y = pca_results.T[1]
    z = pca_results.T[2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=labels)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")


def optimal_cluster_plot(X, n_clusters=10):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from scipy.stats import pearsonr
    # Optimal cluster according to silhouette and elbow method
    plt.figure(figsize=(10, 5))

    inertia = []
    for i in range(1, n_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++')
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_clusters + 1), inertia)
    plt.title('Elbow Method', fontsize=20)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('Inertia', fontsize=20)

    s_score = []
    for i in range(2, n_clusters + 1):  # note that we start from 2: the silhouette is not defined for a single cluster
        kmeans = KMeans(n_clusters=i, init='k-means++')
        labels = kmeans.fit_predict(X)
        s_score.append(silhouette_score(X, labels))

    plt.subplot(1, 2, 2)
    plt.plot(range(2, n_clusters + 1), s_score)
    plt.title('Silhouette', fontsize=20)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('silhouette score', fontsize=20)

    plt.tight_layout()
    plt.show()


def plot_spikes_grouped_by_HU(VH, v, r, fontsize=12):
    # VH.shape = [H, V]

    colors_list = list(colors._colors_full_map.values())
    N_V, T = v.shape
    N_H = r.shape[0]
    stongest_connecting_HU = torch.zeros(N_V)

    for i in range(N_V):
        # returns the index of the strongest connecting HU per visible, according to VH
        stongest_connecting_HU[i] = torch.argmax(torch.abs(VH[:, i]))

    # sort visibles to their strongest connection HU
    idx = torch.argsort(stongest_connecting_HU)
    num = torch.zeros(N_H+1, dtype=int)
    for i in range(N_H):
        # determine how many visibles are connected to hidden i
        num[i+1] = num[i] + torch.count_nonzero(stongest_connecting_HU == i)

    fig, ax = plt.subplots(1, 2, figsize=(12,24))

    sns.heatmap(v[idx,:], ax = ax[1], cbar=False)
    ax[0].set_xlabel('Time', fontsize=fontsize)
    ax[0].set_ylabel('Sorted visible per strongest connecting hidden', fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=15)

    for x, i in enumerate(num):
        ax[0].hlines(i, 0, T, colors = colors_list[x], linewidth=2)

    sns.heatmap(r, ax = ax[2], cbar=False)
    ax[1].set_xlabel('Time', fontsize=fontsize)
    ax[1].set_ylabel('Hidden', fontsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=15)

    plt.show()

def plot_compare_moments(rbm, rtrbm, train_data, test_data, MC_chains=300, chain=50, pre_gibbs_k=50, gibbs_k=20, config_mode=1):

    # catenate train and test data
    a = train_data.shape
    if torch.tensor(a).shape[0] == 3:
        V_train = torch.zeros([a[0], a[1] * a[2]])
        for i in range(a[2]):
            V_train[:, a[1] * i: a[1] * (i + 1)] = train_data[:, :, i]
    elif torch.tensor(a).shape[0] == 2:
        V_train = train_data

    s = test_data.shape
    if torch.tensor(s).shape[0] == 3:
        V_test = torch.zeros([s[0], s[1] * s[2]])
        for i in range(s[2]):
            V_test[:, s[1] * i: s[1] * (i + 1)] = test_data[:, :, i]
    elif torch.tensor(s).shape[0] == 2:
        V_test = test_data

    # create figure
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))

    for i, machine in enumerate([rbm, rtrbm]):
        train_data = train_data.detach().clone().to(machine.device)
        test_data = test_data.detach().clone().to(machine.device)
        V_train = V_train.detach().clone().to(machine.device)
        V_test = V_test.detach().clone().to(machine.device)

        for MC_chain in tqdm(range(MC_chains)):
            if config_mode == 1:
                random_train_config = V_train[:, torch.randint(0, V_train.shape[1], (1,))]
                v_sampled, h_sampled = machine.sample(random_train_config.T, chain=chain, pre_gibbs_k=pre_gibbs_k,
                                                      gibbs_k=gibbs_k, mode=1, disable_tqdm=True)
            elif config_mode == 2:
                # The train and the test set are defined by chopping the data into batches of T=10 or 100 time samples
                # long. The train_data is the first T*train_data_ratio time samples and the test_data is the remaining
                # part T*(1-train_data_ratio) Therefore, we want to initialize our Monte Carlo chain at the last time
                # step of each train_data batch and sample T*(1-train_data_ratio) time steps to compare it to the
                # test_data.
                start_config = train_data[:, -1, MC_chain].view(train_data.shape[0], 1)
                v_sampled, h_sampled = machine.sample(start_config.T, chain=chain, pre_gibbs_k=pre_gibbs_k,
                                                      gibbs_k=gibbs_k, mode=1, disable_tqdm=True)

            if MC_chain == 0:
                v_samples = v_sampled.detach()
                h_samples = h_sampled.detach()
            else:
                v_samples = torch.cat((v_samples, v_sampled), 1)
                h_samples = torch.cat((h_samples, h_sampled), 1)

        if machine is rbm:
            _, H_test = rbm.visible_to_hidden(V_test.T)
            H_test = H_test.T
        elif machine is rtrbm:
            rtrbm_H_test = torch.zeros(rtrbm.N_H, s[1], s[2])
            for j in range(s[2]):
                r = rtrbm.visible_to_expected_hidden(test_data[:, :, j], AF=torch.sigmoid)
                _, rtrbm_H_test[:, :, j] = rtrbm.visible_to_hidden(test_data[:, :, j], r)
            H_test = rtrbm_H_test.view(rtrbm.N_H, s[1] * s[2]).detach()

        means_V_data, means_V_samples, means_H_data, means_H_samples, pw_VV_data, pw_VV_samples, pw_HH_data, \
        pw_HH_samples, pw_VH_data, pw_VH_samples, r_V, r_H, r_VV, r_VH, r_HH = \
            moments(V_test, H_test, v_samples, h_samples)

        ax = axes[0, i]
        xy = np.vstack([means_V_data, means_V_samples])  # Calculate the point density
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(means_V_data[idx], means_V_samples[idx], c=z[idx])
        ax.plot([-1, 1], [-1, 1], ':', color='k')
        ax.set_xlim([0, 1]) #ax.set_xlim([torch.min(means_V_data) - 0.1, torch.max(means_V_data) + 0.1])
        ax.set_ylim([0, 1]) #ax.set_ylim([torch.min(means_V_samples) - 0.1, torch.max(means_V_samples) + 0.1])
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Training data", fontsize=28)
        ax.set_ylabel("Sampled data", fontsize=28)
        ax.set_title("Means V-Units, r-value: {:.2f}".format(r_V), fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax = axes[1, i]
        xy = np.vstack([means_H_data, means_H_samples])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(means_H_data[idx], means_H_samples[idx], c=z[idx])
        ax.plot([-1, 1], [-1, 1], ':', color='k')
        ax.set_xlim([0, 1]) #ax.set_xlim([torch.min(means_H_data) - 0.1, torch.max(means_H_data) + 0.1])
        ax.set_ylim([0, 1]) #ax.set_ylim([torch.min(means_H_samples) - 0.1, torch.max(means_H_samples) + 0.1])
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Training data", fontsize=28)
        ax.set_ylabel("Sampled data", fontsize=28)
        ax.set_title("Means H-Units, r-value: {:.2f}".format(r_H), fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax = axes[2, i]
        xy = np.vstack([pw_VH_data, pw_VH_samples])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(pw_VH_data[idx], pw_VH_samples[idx], c=z[idx])
        ax.plot([-1, 1], [-1, 1], ':', color='k')
        ax.set_xlim([0, 1]) #ax.set_xlim([torch.min(pw_VH_data) - 0.1, torch.max(pw_VH_data) + 0.1])
        ax.set_ylim([0, 1]) # ax.set_ylim([torch.min(pw_VH_samples) - 0.1, torch.max(pw_VH_samples) + 0.1])
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Training data", fontsize=28)
        ax.set_ylabel("Sampled data", fontsize=28)
        ax.set_title("Pairwise moments V/H units, r-value: {:.2f}".format(r_VH), fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax = axes[3, i]
        xy = np.vstack([pw_VV_data, pw_VV_samples])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(pw_VV_data[idx], pw_VV_samples[idx], c=z[idx])
        ax.plot([-1, 1], [-1, 1], ':', color='k')
        ax.set_xlim([0, 1]) #ax.set_xlim([torch.min(pw_VV_data) - 0.1, torch.max(pw_VV_data) + 0.1])
        ax.set_ylim([0, 1]) #ax.set_ylim([torch.min(pw_VV_samples) - 0.1, torch.max(pw_VV_samples) + 0.1])
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Training data", fontsize=28)
        ax.set_ylabel("Sampled data", fontsize=28)
        ax.set_title("Pairwise moments V/V units, r-value: {:.2f}".format(r_VV), fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax = axes[4, i]
        xy = np.vstack([pw_HH_data, pw_HH_samples])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        ax.scatter(pw_HH_data[idx], pw_HH_samples[idx], c=z[idx])
        ax.plot([-1, 1], [-1, 1], ':', color='k')
        ax.set_xlim([0, 1]) #ax.set_xlim([torch.min(pw_HH_data) - 0.1, torch.max(pw_HH_data) + 0.1])
        ax.set_ylim([0, 1]) #ax.set_ylim([torch.min(pw_HH_samples) - 0.1, torch.max(pw_HH_samples) + 0.1])
        # ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Training data", fontsize=28)
        ax.set_ylabel("Sampled data", fontsize=28)
        ax.set_title("Pairwise moments H/H units, r-value: {:.2f}".format(r_HH), fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)

    fig.tight_layout()
    plt.show()

    return


def density_scatter(x , y, ax = None, fig=None, r=None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, s=2, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    ax.plot([0, 1], [0, 1], ':')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.text(.1, .85, 'r-value: {:.2f}'.format(r))

    return ax
