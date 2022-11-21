import torch
import numpy as np
from tqdm import tqdm
from optim.lr_scheduler import get_lrs

import torch
import numpy as np
from tqdm import tqdm
from optim.lr_scheduler import get_lrs


class RBM(torch.nn.Module):
    def __init__(self, W, gamma, theta, b_v):
        super(RBM, self).__init__()
        self.W = W
        self.gamma = gamma
        self.theta = theta
        self.b_v = b_v
        self.parameters = [self.W, self.b_v, self.gamma, self.theta]

    def free_energy(self, v):
        b_v_term = torch.matmul(v, self.b_v)
        cdf = torch.sum((torch.matmul(self.W, v) - self.theta[:, None]) ** 2 / (2 * self.gamma[:, None]) + \
              1 / 2 * torch.log((2 * torch.pi) / self.gamma[:, None]), 0)
        return -cdf - b_v_term

    def pseudo_loglikelihood_rbm(self, v):
        flip_idx = (torch.randint(0, v.shape[0], v.shape[1]), torch.arange(v.shape[1]))
        v_corrupted = v.detach().clone()
        v_corrupted[flip_idx] = 1 - v_corrupted[flip_idx]
        f_true = self.free_energy(v)
        f_corrupted = self.free_energy(v_corrupted)
        return torch.mean(v.shape[0] * torch.log(torch.sigmoid(f_corrupted - f_true)))

    def sample_h_given_v(self, v):
        h = (torch.matmul(self.W, v) - self.theta[:, None]) / self.gamma[:, None] + \
            1 / self.gamma[:, None] * torch.randn(self.gamma.shape)
        if torch.sum(torch.isnan(h))>0 or torch.sum(torch.isin(torch.abs(h))):
            a=1
        return h

    def sample_v_given_h(self, h):
        v_mean = torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v[:, None])
        if torch.sum(torch.isnan(v_mean))>0:
            a=1
        v_sample = torch.bernoulli(v_mean)
        return [v_mean, v_sample]


class ShiftedRBM(RBM):
    def __init__(self, W, gamma, theta, b_v):
        RBM.__init__(self, W=W, gamma=gamma, theta=theta, b_v=b_v)

    def free_energy_given_r_lag(self, v, U, r_lag, b_init, t):
        if t == 0:
            cdf = torch.sum((torch.matmul(self.W, v) - b_init[:, None] - self.theta[:, None]) ** 2 / (2 * self.gamma[:, None]) + \
              0.5 * torch.log((2 * torch.pi) / self.gamma[:, None]), 0)
        else:
            cdf = torch.sum((torch.matmul(self.W, v) - torch.matmul(U, r_lag) - self.theta[:, None]) ** 2 / (2 * self.gamma[:, None]) + \
                  0.5 * torch.log((2 * torch.pi) / self.gamma[:, None]), 0)
        b_v_term = torch.matmul(self.b_v, v)
        return -cdf - b_v_term

    def pseudo_loglikelihood_shifted_rbm(self, v, U, r_lag, b_init, t):
        flip_idx = (np.random.randint(0, v.shape[0], v.shape[1]), np.arange(v.shape[1]))
        v_corrupted = v.detach().clone()
        v_corrupted[flip_idx] = 1 - v_corrupted[flip_idx]
        f_true = self.free_energy_given_r_lag(v, U, r_lag, b_init, t)
        f_corrupted = self.free_energy_given_r_lag(v_corrupted, U, r_lag, b_init, t)
        return torch.mean(v.shape[0] * torch.log(torch.sigmoid(f_corrupted - f_true)))

    def mean_r_lag(self, v, U, r_lag):
        a = (torch.matmul(self.W, v) - torch.matmul(U, r_lag) - self.theta[:, None]) / self.gamma[:, None]
        if torch.sum(torch.isnan(a)) > 0:
            b = 1
        return (torch.matmul(self.W, v) - torch.matmul(U, r_lag) - self.theta[:, None]) / self.gamma[:, None]

    def mean_b_init(self, v, b_init):
        a=(torch.matmul(self.W, v) - b_init[:, None] - self.theta[:, None]) / self.gamma[:, None]
        if torch.sum(torch.isnan(a)) > 0:
            b = 1
        return (torch.matmul(self.W, v) - b_init[:, None] - self.theta[:, None]) / self.gamma[:, None]

    def sample_h_given_v_r_lag(self, v, U, r_lag):
        return self.mean_r_lag(v, U, r_lag) + (1/torch.sqrt(self.gamma) * torch.randn(self.gamma.shape))[:, None]

    def sample_h_given_v_b_init(self, v, b_init):
        return self.mean_b_init(v, b_init) + (1/torch.sqrt(self.gamma) * torch.randn(self.gamma.shape))[:, None]

    def CD_vhv_given_r_lag(self, v_data, U, r_lag, CDk=1):
        h_model = self.sample_h_given_v_r_lag(v_data, U, r_lag)
        for k in range(CDk - 1):
            _, v_model = self.sample_v_given_h(h_model)
            h_model = self.sample_h_given_v_r_lag(v_model, U, r_lag)
            if torch.sum((torch.abs(h_model)>1e4))>0:
                a=1
        _, v_model = self.sample_v_given_h(h_model)
        return [v_model, h_model]

    def CD_vhv_given_b_init(self, v_data, b_init, CDk=1):
        h_model = self.sample_h_given_v_b_init(v_data, b_init)
        for k in range(CDk - 1):
            _, v_model = self.sample_v_given_h(h_model)
            h_model = self.sample_h_given_v_b_init(v_model, b_init)
            if torch.sum((torch.abs(h_model)>1e4))>0:
                a=1
        _, v_model = self.sample_v_given_h(h_model)
        return [v_model, h_model]


class RTRBM(torch.nn.Module):
    def __init__(self, data, n_hidden, batch_size=1, act=torch.sigmoid, device=None):
        super(RTRBM, self).__init__()

        if device is None:
            device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        self.device = device
        self.dtype = torch.float
        self.data = data.detach().clone().to(self.device)  # n_visible, time, n_batches
        self.n_hidden = n_hidden

        if torch.tensor(self.data.shape).shape[0] == 3:
            self.n_batches = data.shape[2] // batch_size
            self.n_visible, self.T, _ = data.shape
        else:
            self.n_batches = 1
            self.n_visible, self.T = data.shape

        self.batch_size = batch_size
        self.activation = act
        self.errors=[]

        self.U = torch.nn.Parameter(0.01 * torch.randn(self.n_hidden, self.n_hidden, dtype=self.dtype, device=self.device))
        self.W = torch.nn.Parameter(0.01 * torch.randn(self.n_hidden, self.n_visible, dtype=self.dtype, device=self.device))
        self.gamma = torch.nn.Parameter(torch.ones(self.n_hidden, dtype=self.dtype, device=self.device))
        self.theta = torch.nn.Parameter(torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device))
        self.b_v = torch.nn.Parameter(torch.zeros(self.n_visible, dtype=self.dtype, device=self.device))
        self.b_init = torch.nn.Parameter(torch.zeros(self.n_hidden, dtype=self.dtype, device=self.device))

        self.r_data = torch.zeros(self.n_hidden, self.T, self.batch_size, dtype=self.dtype, device=self.device)
        self.temporal_layers = []
        for t in range(self.T):
            self.temporal_layers += [ShiftedRBM(W=self.W, gamma=self.gamma, theta=self.theta, b_v=self.b_v)]

    def contrastive_divergence(self, v_data, CDk):
        v_model = torch.zeros(self.n_visible, len(self.temporal_layers), self.batch_size, dtype=self.dtype,
                              device=self.device)
        r_model = torch.zeros(self.n_hidden, len(self.temporal_layers), self.batch_size, dtype=self.dtype,
                              device=self.device)
        self.r_data = self.sample_r_given_v_over_time(v_data)
        if torch.sum(torch.isnan(torch.abs(self.r_data))) > 0:
            a = 1
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                v_model[:, t, :], r_model[:, t, :] = \
                    (layer.CD_vhv_given_b_init(v_data[:, t, :], self.b_init, CDk=CDk))
            else:
                v_model[:, t, :], r_model[:, t, :] = \
                    (layer.CD_vhv_given_r_lag(v_data[:, t, :], self.U, self.r_data[:, t-1, :], CDk=CDk))
        return v_model, r_model

    def sample_r_given_v_over_time(self, v):
        r = torch.zeros(self.n_hidden, self.T, v.shape[2], dtype=self.dtype, device=self.device)
        temp = []
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                r[:, t, :] = layer.mean_b_init(v[:, t, :], self.b_init)
            else:
                r[:, t, :] = layer.mean_r_lag(v[:, t, :], self.U, r[:, t - 1, :])
                temp += [torch.matmul(self.U, r[:, t - 1, :])]
            if torch.sum(torch.abs(r[:, t, :])>1e4) > 0:
                a = 1
        if torch.sum(torch.isnan(r)) > 0:
            a = 1
        return r

    def _log_likelihood_RTRBM(self, v):
        LLH = 0
        for t, layer in enumerate(self.temporal_layers):
            if t == 0:
                r_lag = None
                r = layer.mean_b_init(v[:, t, :], self.b_init)
            else:
                r = layer.mean_r_lag(v[:, t, :], self.U, r_lag)
            LLH += torch.mean(
                self.temporal_layers[t].pseudo_loglikelihood_shifted_rbm(v[:, t, :], self.U, r_lag, self.b_init, t))
            r_lag = r.detach().clone()
        return -LLH

    def learn(self, n_epochs, lr=1e-3, CDk=10, enable_scheduler=False, momentum=0, dampening=0, weight_decay=0, **kwargs):

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True, factor=.5, cooldown=10)

        for epoch in tqdm(range(n_epochs)):
            for batch in range(self.n_batches):
                err = 0
                with torch.no_grad():
                    v_data = self.data[..., batch:batch+self.batch_size].to(self.device)
                    v_model, _ = self.contrastive_divergence(v_data, CDk=CDk)

                loss = self._log_likelihood_RTRBM(v_data) - self._log_likelihood_RTRBM(v_model)
                err += torch.sum((v_data - v_model) ** 2).detach().cpu()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                self.errors += [err / self.data.numel()]
                if enable_scheduler:
                    scheduler.step(self.errors[-1])
        return
    def infer(self, data, pre_gibbs_k=50, gibbs_k=10, mode=2, t_extra=0, disable_tqdm=False):

        T = self.T
        n_v, t1 = data.shape

        vt = torch.zeros(n_v, T + t_extra, dtype=self.dtype, device=self.device)
        rt = torch.zeros(self.n_hidden, T + t_extra, dtype=self.dtype, device=self.device)
        vt[:, 0:t1] = data.float().to(self.device)

        rt[:, 0] = self.activation(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, t1):
            rt[:, t] = self.activation(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        for t in tqdm(range(t1, T + t_extra), disable=disable_tqdm):
            v = vt[:, t - 1]

            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(self.activation(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(self.activation(torch.matmul(self.W.T, h) + self.b_v.T))

            vt_k = torch.zeros(n_v, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.n_hidden, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(self.activation(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(self.activation(torch.matmul(self.W.T, h) + self.b_v.T))
                vt_k[:, kk] = v.T
                ht_k[:, kk] = h.T

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
            if mode == 3:
                E = torch.sum(ht_k * (torch.matmul(self.W, vt_k)), 0) + torch.matmul(self.b_v, vt_k) + torch.matmul(
                    self.b_h, ht_k) + torch.matmul(torch.matmul(self.U, rt[:, t - 1]).T, ht_k)
                idx = torch.argmax(E)
                vt[:, t] = vt_k[:, idx]

            rt[:, t] = self.activation(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        return vt, rt

class apply_constraints(object):

    def __init__(self, gamma_):
        self.gamma_ = gamma_

    def __call__(self, module, gamma_min=torch.tensor(0.05), gamma_max_deviate=torch.tensor(1)):
        if hasattr(module, 'gamma'):
            gamma = module.gamma.data
            gamma = torch.maximum(gamma_min,
                                  torch.clamp_(gamma, min=torch.maximum((1-gamma_max_deviate) * self.gamma_, gamma_min)
                                               , max=(1+gamma_max_deviate) * self.gamma_))

if __name__ == '__main__':
    import os

    # os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')
    from data.mock_data import create_BB
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    data = create_BB(N_V=16, T=320, n_samples=10, width_vec=[4, 5, 6], velocity_vec=[1, 2])
    rtrbm = RTRBM(data=data, n_hidden=8, batch_size=10, device='cpu')

    # rtrbm.learn(n_epochs=100, CDk=10, lr_schedule='geometric_decay', max_lr=1e-2, min_lr=1e-3)
    rtrbm.learn(n_epochs=100, CDk=1, lr=1e-3)

    # Infer from trained RTRBM and plot some results
    vt_infer, rt_infer = rtrbm.infer(torch.tensor(data[:, :280 // 2, 0]))

    # effective coupling
    W = rtrbm.W.detach().numpy()
    U = rtrbm.U.detach().numpy()
    rt = rtrbm.rt.detach().numpy()
    data = data.detach().numpy()
    var_h_matrix = np.reshape(np.var(rt[..., 0], 1).repeat(W.shape[1]), [W.shape[1], W.shape[0]]).T
    var_v_matrix = np.reshape(np.var(data[..., 0], 1).repeat(W.shape[0]), [W.shape[0], W.shape[1]])

    Je_Wv = np.matmul(W.T, W * var_h_matrix) / W.shape[1] ** 2
    Je_Wh = np.matmul(W * var_v_matrix, W.T) / W.shape[0] ** 2

    _, ax = plt.subplots(2, 3, figsize=(12, 12))
    sns.heatmap(vt_infer[:, 270:].detach().numpy(), ax=ax[0, 0], cbar=False)
    ax[0, 0].set_title('Infered data')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Neuron index')

    ax[0, 1].plot(rtrbm.err)
    ax[0, 1].set_title('RMSE of the RTRBM over epoch')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('RMSE')

    sns.heatmap(Je_Wv, ax=ax[0, 2])
    ax[0, 2].set_title('Effective coupling V')
    ax[0, 2].set_xlabel("Visibel nodes")
    ax[0, 2].set_ylabel("Visibel nodes")

    sns.heatmap(rtrbm.W.detach().numpy(), ax=ax[1, 0])
    ax[1, 0].set_title('Visible to hidden connection')
    ax[1, 0].set_xlabel('Visible')
    ax[1, 0].set_ylabel('Hiddens')

    sns.heatmap(rtrbm.U.detach().numpy(), ax=ax[1, 1])
    ax[1, 1].set_title('Hidden to hidden connection')
    ax[1, 1].set_xlabel('Hidden(t-1)')
    ax[1, 1].set_ylabel('Hiddens(t)')

    sns.heatmap(Je_Wh, ax=ax[1, 2])
    ax[1, 2].set_title('Effective coupling H')
    ax[1, 2].set_xlabel("Hidden nodes [t]")
    ax[1, 2].set_ylabel("Hidden nodes [t]")
    plt.show()

