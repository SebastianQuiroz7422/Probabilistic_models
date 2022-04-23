import torch
import numpy as np
from tqdm import tqdm
from boltzmann_machines.data.reshape_data import reshape_from_batches, train_test_split
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from matplotlib import cm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from boltzmann_machines.data import load_data


def correlations(results):
    vt, vs, ht, hs = results
    vvt, vvs, vht, vhs, hht, hhs = calculate_moments(vt, ht, vs, hs)
    r_v, _ = pearsonr(torch.mean(vt, 1), torch.mean(vs, 1))
    r_h, _ = pearsonr(torch.mean(ht, 1), torch.mean(hs, 1))
    r_vv, _ = pearsonr(vvt, vvs)
    r_vh, _ = pearsonr(vht, vhs)
    r_hh, _ = pearsonr(hht, hhs)
    return r_v, r_h, r_vv, r_vh, r_hh


def infer_and_get_moments_plot(dir,
                               test=None,
                               pre_gibbs_k=0, gibbs_k=1, mode=1,
                               n_batches=None, n=1000, m=50000,
                               machine='rtrbm', plot=True,
                               ax=None, fig=None):

    rtrbm = torch.load(dir, map_location='cpu')
    rtrbm.device = 'cpu'

    if test is None:
        test = rtrbm.V.clone().detach()
    T = test.shape[1]
    if n_batches is None:
        n_batches = test.shape[2]
    vt = test.clone().detach()
    ht = torch.empty(rtrbm.N_H, T, n_batches)
    vs = torch.empty(rtrbm.N_V, T, n_batches)
    hs = torch.empty(rtrbm.N_H, T, n_batches)

    for i in tqdm(range(n_batches)):
        if machine == 'rtrbm':
            rt = rtrbm.visible_to_expected_hidden(test[:, :, i])
            x, _ = rtrbm.visible_to_hidden(test[:, :, i], rt)
            ht[:, :, i] = x.clone().detach()
        elif machine == 'rbm':
            x, _ = rtrbm.visible_to_hidden(test[:, :, i])
            ht[:, :, i] = x.clone().detach()
        else:
            raise ValueError('Machine must be "rbm" or "rtrbm"')

        v, h = rtrbm.sample(test[:, 0, i], chain=test.shape[1], pre_gibbs_k=pre_gibbs_k,
                            gibbs_k=gibbs_k, mode=mode, disable_tqdm=True)
        vs[:, :, i] = v.clone().detach().cpu()
        hs[:, :, i] = h.clone().detach().cpu()

    vt = reshape_from_batches(vt)
    ht = reshape_from_batches(ht)
    vs = reshape_from_batches(vs)
    hs = reshape_from_batches(hs)

    vvt, vvs, vht, vhs, hht, hhs = calculate_moments(vt, ht, vs, hs, n=n, m=m)

    vt_mean = np.mean(np.array(vt), axis=1)
    vs_mean = np.mean(np.array(vs), axis=1)
    ht_mean = np.mean(np.array(ht), axis=1)
    hs_mean = np.mean(np.array(hs), axis=1)

    if plot:
        ax, fig = density_plot_moments(vt_mean, vs_mean, ht_mean, hs_mean, vvt, vvs, hht, hhs, vht, vhs, ax=ax, fig=fig)
        return ax, fig, [vt, vs, ht, hs]

    elif plot==False:
        r2v = np.corrcoef(vt_mean, vs_mean)[1, 0]**2
        r2v2 = np.corrcoef(vvt, vvs)[1, 0]**2
        r2h = np.corrcoef(ht_mean, hs_mean)[1, 0]**2
        r2h2 = np.corrcoef(hht, hhs)[1, 0]**2
        r2vh = np.corrcoef(vht, vhs)[1, 0]**2
        return [vt, vs, ht, hs], [r2v, r2v2, r2h, r2h2, r2vh]


def calculate_moments(vt, ht, vs, hs, n=1000, m=50000):
    if vt.shape[0] > n:
        idx = torch.randperm(vt.shape[0])[:n]
        vt = vt[idx, :]
        vs = vs[idx, :]

    vvt = np.array(torch.matmul(vt, vt.T) / vt.shape[1]).flatten()
    vvs = np.array(torch.matmul(vs, vs.T) / vs.shape[1]).flatten()
    vht = np.array(torch.matmul(vt, ht.T) / vt.shape[1]).flatten()
    vhs = np.array(torch.matmul(vs, hs.T) / vs.shape[1]).flatten()
    hht = np.array(torch.matmul(ht, ht.T) / ht.shape[1]).flatten()
    hhs = np.array(torch.matmul(hs, hs.T) / hs.shape[1]).flatten()

    if vvt.shape[0] > m:
        idx = torch.randperm(vvt.shape[0])[:m]
        vvt = vvt[idx]
        vvs = vvs[idx]
    if vht.shape[0] > m:
        idx = torch.randperm(vht.shape[0])[:m]
        vht = vht[idx]
        vhs = vhs[idx]
    if hht.shape[0] > m:
        idx = torch.randperm(hht.shape[0])[:m]
        hht = hht[idx]
        hhs = hhs[idx]

    return vvt, vvs, vht, vhs, hht, hhs


def density_scatter(x, y, ax=None, fig=None, r=None, sort=True, bins=20, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    x[np.isnan(x)] = 0
    y[np.isnan(y)] = 0
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T,
                method="splinef2d", bounds_error=False)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, s=2, **kwargs )

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    ax.plot([0, 1], [0, 1], ':')
    ax.set_xlim([0, np.max(x)])
    ax.set_ylim([0, np.max(y)])
    ax.set_xticks([0, np.floor(10 * np.max(x)) / 10])
    ax.set_yticks([0, np.floor(10 * np.max(y)) / 10])

    if r is None:
        r, _ = pearsonr(x, y)
    ax.text(.1, .9, 'r-value: {:.2f}'.format(r), transform=ax.transAxes)

    return ax


def density_plot_moments(vt_mean, vs_mean, ht_mean, hs_mean, vvt, vvs, hht, hhs, vht, vhs, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(1, 5, figsize=(30, 4))
    density_scatter(vt_mean, vs_mean, ax=ax[0], fig=fig)
    density_scatter(ht_mean, hs_mean, ax=ax[1], fig=fig)
    density_scatter(vvt, vvs, ax=ax[2], fig=fig)
    density_scatter(hht, hhs, ax=ax[3], fig=fig)
    density_scatter(vht, vhs, ax=ax[4], fig=fig)
    return ax, fig


if __name__ == '__main__':
    #ax, res = infer_and_get_moments_plot('../data/part brain/1000 neurons/rbm.pt', n=10000, machine='rbm')
    _, _, vh = infer_and_get_moments_plot(dir=r'C:\Users\sebas\RU\intern\Figures\data\rtrbm_3_pop_PMT', n_batches=100,
                                          pre_gibbs_k=100, gibbs_k=100, mode=1, n=1000, m=50000)
    plt.show()
