"""
This is a practical file for testing quick ideas. Sebastian, please don't adjust this one :)
"""

import torch
import numpy as np
from tqdm import tqdm
from optim.lr_scheduler import get_lrs


class RTRBM(object):
    def __init__(self, data, N_H=10, device='cuda', no_bias=False, debug_mode=False):
        if not torch.cuda.is_available():
            print('cuda not available, using cpu')
            self.device = 'cpu'
        else:
            self.device = device
        self.dtype = torch.float
        self.V = data
        self.dim = torch.tensor(self.V.shape).shape[0]
        if self.dim == 2:
            self.N_V, self.T = self.V.shape
            self.num_samples = 1
        elif self.dim == 3:
            self.N_V, self.T, self.num_samples = self.V.shape
        else:
            raise ValueError("Data is not correctly defined: Use (N_V, T) or (N_V, T, num_samples) dimensions")
        self.N_H = N_H
        self.W = 0.01/self.N_V * torch.randn(self.N_H, self.N_V, dtype=self.dtype, device=self.device)
        self.U = 0.01/self.N_H * torch.randn(self.N_H, self.N_H, dtype=self.dtype, device=self.device)
        self.b_H = torch.zeros(1, self.N_H, dtype=self.dtype, device=self.device)
        self.b_V = torch.zeros(1, self.N_V, dtype=self.dtype, device=self.device)
        self.b_init = torch.zeros(1, self.N_H, dtype=self.dtype, device=self.device)
        self.params = [self.W, self.U, self.b_H, self.b_V, self.b_init]
        self.no_bias = no_bias
        self.debug_mode = debug_mode
        if debug_mode:
            self.parameter_history = []

        self.Dparams = self.initialize_grad_updates()
        self.errors = []

    def learn(self, n_epochs=1000,
              lr=None,
              lr_schedule=None,
              batch_size=1,
              CDk=10,
              PCD=False,
              sp=None, x=2,
              mom=0.9, wc=0.0002,
              AF=torch.sigmoid,
              disable_tqdm=False,
              save_every_n_epochs=1, shuffle_batch=True,
              **kwargs):

        if self.dim == 2:
            num_batches = 1
            batch_size = 1
        elif self.dim == 3:
            num_batches = self.num_samples // batch_size
        if lr is None:
            lrs = np.array(get_lrs(mode=lr_schedule, n_epochs=n_epochs, **kwargs))
        else:
            lrs = lr * torch.ones(n_epochs)

        self.disable = disable_tqdm
        self.lrs = lrs
        for epoch in tqdm(range(0, n_epochs), disable=self.disable):
            err = 0
            for batch in range(0, num_batches):
                self.dparams = self.initialize_grad_updates()
                for i in range(0, batch_size):
                    if self.dim == 2:
                        vt = self.V.to(self.device)
                    elif self.dim == 3:
                        vt = self.V[:, :, batch * batch_size + i].to(self.device)
                    rt = self.visible_to_expected_hidden(vt, AF=AF)
                    if PCD and epoch != 0:
                        barht, barvt, ht_k, vt_k = self.CD(vt_k[:, :, -1], rt, CDk, AF=AF)
                    else:
                        barht, barvt, ht_k, vt_k = self.CD(vt, rt, CDk, AF=AF)
                    err += torch.sum((vt - vt_k[:, :, -1]) ** 2).cpu()
                    dparam = self.grad(vt, rt, ht_k, vt_k, barvt, barht, CDk)
                    for i in range(len(dparam)): self.dparams[i] += dparam[i] / batch_size
                self.update_grad(lr=lrs[epoch], mom=mom, wc=wc, sp=sp, x=x)
            self.errors += [err / self.V.numel()]
            if self.debug_mode and epoch % save_every_n_epochs == 0:
                self.parameter_history.append([param.detach().clone().cpu() for param in self.params])
            if shuffle_batch:
                self.V[..., :] = self.V[..., torch.randperm(self.num_samples)]
        self.r = rt

    def return_params(self):
        return [self.W, self.U, self.b_V, self.b_init, self.b_H, self.errors]

    def CD(self, vt, rt, CDk, AF=torch.sigmoid):
        ht_k = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        probht_k = torch.zeros(self.N_H, self.T, CDk, dtype=self.dtype, device=self.device)
        vt_k = torch.zeros(self.N_V, self.T, CDk, dtype=self.dtype, device=self.device)
        vt_k[:, :, 0] = vt.detach()
        probht_k[:, :, 0], ht_k[:, :, 0] = self.visible_to_hidden(vt_k[:, :, 0], rt, AF=AF)
        for kk in range(1, CDk):
            vt_k[:, :, kk] = self.hidden_to_visible(ht_k[:, :, 0], AF=AF)
            probht_k[:, :, kk], ht_k[:, :, kk] = self.visible_to_hidden(vt_k[:, :, kk], rt, AF=AF)
        barht = torch.mean(probht_k, 2)
        barvt = torch.mean(vt_k, 2)
        return barht, barvt, ht_k, vt_k

    def visible_to_expected_hidden(self, vt, AF=torch.sigmoid):
        T = vt.shape[1]
        rt = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        rt[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, T):
            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_H + torch.matmul(self.U, rt[:, t - 1]))
        return rt

    def visible_to_hidden(self, vt, r, AF=torch.sigmoid):
        T = vt.shape[1]
        ph_sample = torch.zeros(self.N_H, T, dtype=self.dtype, device=self.device)
        ph_sample[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        ph_sample[:, 1:T] = AF(torch.matmul(self.W, vt[:, 1:T]).T + torch.matmul(self.U, r[:, 0:T - 1]).T + self.b_H).T
        return ph_sample, torch.bernoulli(ph_sample)

    def hidden_to_visible(self, h, AF=torch.sigmoid):
        return torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))

    def initialize_grad_updates(self):
        return [torch.zeros_like(param, dtype=self.dtype, device=self.device) for param in self.params]

    def grad(self, vt, rt, ht_k, vt_k, barvt, barht, CDk):
        Dt = torch.zeros(self.N_H, self.T + 1, dtype=self.dtype, device=self.device)
        for t in range(self.T - 1, -1, -1):
            Dt[:, t] = torch.matmul(self.U.T, (Dt[:, t + 1] * rt[:, t] * (1 - rt[:, t]) + (rt[:, t] - barht[:, t])))
        db_init = (rt[:, 0] - barht[:, 0]) + Dt[:, 1] * rt[:, 0] * (1 - rt[:, 0])
        tmp = torch.sum(Dt[:, 2:self.T] * (rt[:, 1:self.T - 1] * (1 - rt[:, 1:self.T - 1])), 1)
        db_H = torch.sum(rt[:, 1:self.T], 1) - torch.sum(barht[:, 1:self.T], 1) + tmp
        db_V = torch.sum(vt - barvt, 1)
        dW_1 = torch.sum((Dt[:, 1:self.T] * rt[:, 0:self.T - 1] * (1 - rt[:, 0:self.T - 1])).unsqueeze(1).repeat(1, self.N_V, 1) * vt[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)
        dW_2 = torch.sum(rt.unsqueeze(1).repeat(1, self.N_V, 1) * vt.unsqueeze(0).repeat(self.N_H, 1, 1), 2) - torch.sum(torch.sum(ht_k.unsqueeze(1).repeat(1, self.N_V, 1, 1) * vt_k.unsqueeze(0).repeat(self.N_H, 1, 1, 1),3), 2) / CDk
        dW = dW_1 + dW_2
        dU = torch.sum((Dt[:, 2:self.T + 1] * (rt[:, 1:self.T] * (1 - rt[:, 1:self.T])) + rt[:, 1:self.T] - barht[:, 1:self.T]).unsqueeze(1).repeat(1, self.N_H, 1) * rt[:, 0:self.T - 1].unsqueeze(0).repeat(self.N_H, 1, 1), 2)
        return [dW, dU, db_H, db_V, db_init]

    def update_grad(self, lr=1e-3, mom=0, wc=0, x=2, sp=None):
        dW, dU, db_H, db_V, db_init = self.dparams
        DW, DU, Db_H, Db_V, Db_init = self.Dparams
        DW = mom * DW + lr * (dW - wc * self.W)
        DU = mom * DU + lr * (dU - wc * self.U)
        Db_H = mom * Db_H + lr * db_H
        Db_V = mom * Db_V + lr * db_V
        Db_init = mom * Db_init + lr * db_init
        if sp is not None:
            DW -= sp * torch.reshape(torch.sum(torch.abs(self.W), 1).repeat(self.N_V), [self.N_H, self.N_V]) ** (x - 1) * torch.sign(self.W)
        if self.no_bias:
            Db_H, Db_V, Db_init = 0, 0, 0
        self.Dparams = [DW, DU, Db_H, Db_V, Db_init]
        for i in range(len(self.params)): self.params[i] += self.Dparams[i]
        return

    def infer(self,
              data,
              AF=torch.sigmoid,
              pre_gibbs_k=50,
              gibbs_k=10,
              mode=2,
              t_extra=0,
              disable_tqdm=False):

        T = self.T
        N_H = self.N_H
        N_V, t1 = data.shape

        vt = torch.zeros(N_V, T + t_extra, dtype=self.dtype, device=self.device)
        rt = torch.zeros(N_H, T + t_extra, dtype=self.dtype, device=self.device)
        vt[:, 0:t1] = data.float().to(self.device)

        rt[:, 0] = AF(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, t1):
            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_H + torch.matmul(self.U, rt[:, t - 1]))

        for t in tqdm(range(t1, T + t_extra), disable=disable_tqdm):
            v = vt[:, t - 1]

            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_H + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))

            vt_k = torch.zeros(N_V, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(N_H, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_H + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))
                vt_k[:, kk] = v.T
                ht_k[:, kk] = h.T

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
            if mode == 3:
                E = torch.sum(ht_k * (torch.matmul(self.W, vt_k)), 0) + torch.matmul(self.b_V, vt_k) + torch.matmul(
                    self.b_H, ht_k) + torch.matmul(torch.matmul(self.U, rt[:, t - 1]).T, ht_k)
                idx = torch.argmax(E)
                vt[:, t] = vt_k[:, idx]

            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_H + torch.matmul(self.U, rt[:, t - 1]))

        return vt, rt

    def sample(self, v_start, AF=torch.sigmoid, chain=50, pre_gibbs_k=100, gibbs_k=20, mode=1, disable_tqdm=False):
        vt = torch.zeros(self.N_V, chain + 1, dtype=self.dtype, device=self.device)
        rt = torch.zeros(self.N_H, chain + 1, dtype=self.dtype, device=self.device)
        rt[:, 0] = AF(torch.matmul(self.W, v_start.T) + self.b_init)
        vt[:, 0] = v_start
        for t in tqdm(range(1, chain + 1), disable=disable_tqdm):
            v = vt[:, t - 1]
            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_H + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))

            vt_k = torch.zeros(self.N_V, gibbs_k, dtype=self.dtype, device=self.device)
            ht_k = torch.zeros(self.N_H, gibbs_k, dtype=self.dtype, device=self.device)
            for kk in range(gibbs_k):
                h = torch.bernoulli(AF(torch.matmul(self.W, v).T + self.b_H + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(AF(torch.matmul(self.W.T, h) + self.b_V.T))
                vt_k[:, kk] = v.T
                ht_k[:, kk] = h.T

            if mode == 1:
                vt[:, t] = vt_k[:, -1]
            if mode == 2:
                vt[:, t] = torch.mean(vt_k, 1)
            if mode == 3:
                E = torch.sum(ht_k * (torch.matmul(self.W, vt_k)), 0) + torch.matmul(self.b_V, vt_k) + torch.matmul(
                    self.b_H, ht_k) + torch.matmul(torch.matmul(self.U, rt[:, t - 1]).T, ht_k)
                idx = torch.argmax(E)
                vt[:, t] = vt_k[:, idx]

            rt[:, t] = AF(torch.matmul(self.W, vt[:, t]) + self.b_H + torch.matmul(self.U, rt[:, t - 1]))

        return vt[:, 1:], rt[:, 1:]

if __name__ == '__main__':
    import os

    # os.chdir(r'D:\OneDrive\RU\Intern\rtrbm_master')
    from boltzmann_machines.data import create_BB
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    data = create_BB(N_V=16, T=320, n_samples=10, width_vec=[4, 5, 6], velocity_vec=[1, 2])

    rtrbm = RTRBM(data, N_H=8, device="cpu")
    rtrbm.learn(batchsize=10, n_epochs=1000, lr=1e-3, lr_mode=None, CDk=10, mom=0, wc=0, sp=0, x=0)


    # Infer from trained RTRBM and plot some results
    vt_infer, rt_infer = rtrbm.infer(torch.tensor(data[:, :280 // 2, 0]))

    # effective coupling
    W = rtrbm.W.detach().clone().numpy()
    U = rtrbm.U.detach().clone().numpy()
    rt = rtrbm.r.detach().clone().numpy()[..., None]
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

    ax[0, 1].plot(rtrbm.errors)
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
