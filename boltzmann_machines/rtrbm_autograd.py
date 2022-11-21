import torch
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RBM(torch.nn.Module):

    def __init__(self, n_hidden: int, n_visible: int):
        super().__init__()

        W = 0.01 * torch.randn(n_hidden, n_visible, dtype=torch.float64)
        b_h = torch.zeros(n_visible, dtype=torch.float64)
        b_v = torch.zeros(n_hidden, dtype=torch.float64)

        self.W = torch.nn.Parameter(W)
        self.b_h = torch.nn.Parameter(b_h)
        self.b_v = torch.nn.Parameter(b_v)

    def init_with_specified_parameters(self, W: torch.Tensor = None, b_h: torch.Tensor = None, b_v: torch.Tensor = None):
        if W is not None:
            self.W = torch.nn.Parameter(W)
        if b_h is not None:
            self.b_h = torch.nn.Parameter(b_h)
        if b_v is not None:
            self.b_v = torch.nn.Parameter(b_v)

    def free_energy(self, v: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        b_v_term = torch.matmul(self.b_v, v)
        cgf = torch.sum(torch.log(1 + torch.exp(torch.matmul(beta * self.W, v) + self.b_h[:, None])), 0)
        return -cgf - b_v_term

    def pseudo_loglikelihood(self, v: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        flip_idx = (torch.randint(0, v.shape[0], v.shape[1]), torch.arange(v.shape[1]))
        v_corrupted = v.detach().clone()
        v_corrupted[flip_idx] = 1 - v_corrupted[flip_idx]
        f_true = self.free_energy(v, beta=beta)
        f_corrupted = self.free_energy(v_corrupted, beta=beta)
        return torch.mean(v.shape[0] * torch.log(torch.sigmoid(f_corrupted - f_true)))

    def _sample_h_given_v(self, v: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        h_mean = torch.sigmoid(beta * (torch.matmul(self.W, v) + self.b_h[:, None]))
        h_sample = torch.bernoulli(h_mean)
        return h_sample

    def _sample_v_given_h(self, h: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        v_mean = torch.sigmoid(beta * (torch.matmul(self.W.T, h) + self.b_v[:, None]))
        v_sample = torch.bernoulli(v_mean)
        return v_sample

    def forward(self, v_data: torch.Tensor, CDk: int = 1, beta: float = 1.0) -> torch.Tensor:
        h_model = self._sample_h_given_v(v_data)
        for i in range(CDk - 1):
            v_model = self._sample_v_given_h(h_model)
            h_model = self._sample_h_given_v(v_model)
        v_model = self._sample_v_given_h(h_model)
        return v_model

class ShiftedRBM(RBM):
    def __init__(self, n_hidden: int, n_visible: int):
        super().__init__(n_hidden=n_hidden, n_visible=n_visible)

        U = 0.01 * torch.randn(n_hidden, n_hidden, dtype=torch.float64)
        self.U = torch.nn.Parameter(U)

    def init_with_specified_parameters(self, W: torch.Tensor = None, U: torch.Tensor = None, b_h: torch.Tensor = None,
                                       b_v: torch.Tensor = None):
        if W is not None:
            self.W = torch.nn.Parameter(W)
        if U is not None:
            self.U = torch.nn.Parameter(U)
        if b_h is not None:
            self.b_h = torch.nn.Parameter(b_h)
        if b_v is not None:
            self.b_v = torch.nn.Parameter(b_v)

    def free_energy(self, v: torch.Tensor, r_lag: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        cgf = torch.sum(torch.log(1 + torch.exp(beta * (torch.matmul(self.W, v) + torch.matmul(self.U, r_lag) + self.b_h[:, None]))), 0)
        b_v_term = torch.matmul(self.b_v, v)
        return -cgf - b_v_term

    def pseudo_loglikelihood(self, v: torch.Tensor, r_lag: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        flip_idx = (torch.randint(0, v.shape[0], (v.shape[1],)), torch.arange(v.shape[1]))
        v_corrupted = v.detach().clone()
        v_corrupted[flip_idx] = 1 - v_corrupted[flip_idx]
        f_true = self.free_energy(v, r_lag, beta=beta)
        f_corrupted = self.free_energy_given(v_corrupted, r_lag, beta=beta)
        return torch.mean(v.shape[0] * torch.log(torch.sigmoid(f_corrupted - f_true)))

    def _sample_h_given_v_r_lag(self, v: torch.Tensor, r_lag: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        r = torch.sigmoid(beta * (torch.matmul(self.W, v) + torch.matmul(self.U, r_lag) + self.b_h[:, None]))
        return r, torch.bernoulli(r)

    def forward(self, v_data: torch.Tensor, r: torch.Tensor, CDk: int = 1, beta: float = 1.0) -> torch.Tensor:
        r_data, h_model = self.sample_h_given_v_r_lag(v_data, self.U, r, beta=beta)
        for k in range(CDk - 1):
            v_model = self._sample_v_given_h(h_model, beta=beta)
            _, h_model = self._sample_h_given_v_r_lag(v_model, self.U, r_data, beta=beta)
        v_model = self.sample_v_given_h(h_model, beta=beta)
        return v_model


class RTRBM(torch.nn.Module):
    def __init__(self, n_hidden, n_visible, T):
        super().__init__()

        self.W = torch.nn.Parameter(torch.empty(n_hidden, n_visible, dtype=torch.float))
        self.U = torch.nn.Parameter(torch.empty(n_hidden, n_hidden, dtype=torch.float))
        self.b_h = torch.nn.Parameter(torch.empty(n_hidden, dtype=torch.float))
        self.b_init = torch.nn.Parameter(torch.empty(n_hidden, dtype=torch.float))
        self.b_v = torch.nn.Parameter(torch.empty(n_visible, dtype=torch.float))
        self._init_params()

        self.temporal_layers = []
        for t in range(T):
            if t == 0:
                shifted_rbm = ShiftedRBM(n_hidden=n_hidden, n_visible=n_visible)
                shifted_rbm.init_with_specified_parameters(self.W, torch.zeros(n_hidden, n_hidden), self.b_init, self.b_v)
            else:
                shifted_rbm = ShiftedRBM(n_hidden=n_hidden, n_visible=n_visible)
                shifted_rbm.init_with_specified_parameters(self.W, self.U, self.b_h, self.b_v)
            self.temporal_layers += [shifted_rbm]

        self.temporal_layers = torch.nn.ModuleList(self.temporal_layers)
        self.n_hidden = n_hidden

    def _init_params(self):
        torch.nn.init.xavier_uniform(self.W, gain=torch.nn.init.calculate_gain('sigmoid'))
        torch.nn.init.xavier_uniform(self.U, gain=torch.nn.init.calculate_gain('sigmoid'))
        torch.nn.init.zeros_(self.b_h)
        torch.nn.init.zeros_(self.b_init)
        torch.nn.init.zeros_(self.b_v)

    def _parallel_log_Z_AIS(self, v: torch.Tensor, r: torch.Tensor, n_betas: int = 100) -> torch.Tensor:# annealed_importance_sampling
        betas = torch.linspace(0, 1, n_betas)
        log_Z_init = torch.sum(self._parallel_cumulant_generative_function(v, r, beta=betas[0]), 0)
        log_weights = torch.zeros(v.shape[1], v.shape[2])
        for i in range(1, n_betas):
            energy = -self._parallel_free_energy(v, r, beta=betas[i-1])
            log_weights += -(betas[i] - betas[i - 1]) * energy
            v, _, _ = self.forward(v, CDk=1, beta=betas[i])
        logZ = torch.mean(log_Z_init + log_weights)
        logZ_std = torch.std(log_Z_init + log_weights)
        return log_Z_init + log_weights

    def _parallel_AIS_loglikelihood_AIS(self, v: torch.Tensor, r: torch.Tensor, n_betas: int = 100) -> torch.Tensor:
        return -torch.mean(torch.sum(self._parallel_free_energy(v, r) + self._parallel_log_Z_AIS(v, r, n_betas), 0))

    def _parallel_cumulant_generative_function(self, v: torch.Tensor, r: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        exp_arg = torch.einsum('hv, vb->hb', self.W, v[:, 0, :]) + self.b_init[:, None]
        exp_arg = torch.cat([exp_arg[:, None, :], torch.einsum('hv, vTb->hTb', self.W, v[:, 1:, :]) +
                             torch.einsum('hr, rTb->hTb', self.U, r[:, :-1, :]) + self.b_h[:, None, None]], 1)
        return torch.log(1 + torch.exp(beta * exp_arg))

    def _parallel_free_energy(self, v: torch.Tensor, r: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        return -torch.sum(self._parallel_cumulant_generative_function(v, r, beta=beta), 0) - beta * torch.einsum('v, vTb->Tb', self.b_v, v)

    def _parallel_pseudo_loglikelihood(self, v: torch.Tensor, r: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        flip_idx = (torch.randint(0, v.shape[0], (v.shape[1]*v.shape[2],)), torch.arange(v.shape[1]*v.shape[2]))
        v_corrupted = v.detach().clone().reshape(v.shape[0], v.shape[1]*v.shape[2])
        v_corrupted[flip_idx] = 1 - v_corrupted[flip_idx]
        v_corrupted = v_corrupted.reshape(v.shape[0], v.shape[1], v.shape[2])
        f_true = self._parallel_free_energy(v, r, beta=beta)
        f_corrupted = self._parallel_free_energy(v_corrupted, r, beta=beta)
        return -torch.sum(torch.mean(v.shape[0] * torch.log(torch.sigmoid(f_corrupted - f_true)), 1))

    def _parallel_sample_r_h_given_v(self, v: torch.Tensor, r: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        r0 = torch.sigmoid(beta * (torch.einsum('hv, vb->hb', self.W, v[:, 0, :]) + self.b_init[:, None]))
        r = torch.sigmoid(beta * (torch.einsum('hv, vTb->hTb',self.W, v[:, 1:, :]) +
                          torch.einsum('hr, rTb->hTb', self.U, r[:, :-1, :]) + self.b_h[:, None, None]))
        r = torch.cat([r0[:, None, :], r], 1)
        h = torch.bernoulli(r)
        return r, h

    def _parallel_sample_v_given_h(self, h: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        v_prob = torch.sigmoid(beta * (torch.einsum('hv, hTb->vTb', self.W, h) + self.b_v[:, None, None]))
        return torch.bernoulli(v_prob)

    def _sample_r_given_v_over_time(self, v: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        r = torch.zeros(self.n_hidden, v.shape[1], v.shape[2], device=v.device)
        r_lag = torch.zeros(self.n_hidden, v.shape[2])
        for t, layer in enumerate(self.temporal_layers):
            r[:, t, :] = layer._sample_h_given_v_r_lag(v[:, t, :], r_lag, beta=beta)[0]
            r_lag = r[:, t, :]
        return r

    def forward(self, v_data: torch.Tensor, CDk: int = 1, beta: float = 1.0) -> torch.Tensor:
        r_data = self._sample_r_given_v_over_time(v_data, beta=beta)
        _, h_model = self._parallel_sample_r_h_given_v(v_data, r_data, beta=beta)
        for i in range(CDk - 1):
            v_model = self._parallel_sample_v_given_h(h_model, beta=beta)
            _, h_model = self._parallel_sample_r_h_given_v(v_model, r_data, beta=beta)
        v_model = self._parallel_sample_v_given_h(h_model, beta=beta)
        r_model = self._sample_r_given_v_over_time(v_model, beta=beta)
        return v_model, r_data, r_model

    def infer(self, data, pre_gibbs_k=50, gibbs_k=10, mode=2, t_extra=0, disable_tqdm=False):

        T = len(self.temporal_layers)
        n_v, t1 = data.shape
        n_h = self.W.shape[0]
        vt = torch.zeros(n_v, T + t_extra, dtype=torch.float)
        rt = torch.zeros(n_h, T + t_extra, dtype=torch.float)
        vt[:, 0:t1] = data.float()

        rt[:, 0] = torch.sigmoid(torch.matmul(self.W, vt[:, 0]) + self.b_init)
        for t in range(1, t1):
            rt[:, t] = torch.sigmoid(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        for t in tqdm(range(t1, T + t_extra), disable=disable_tqdm):
            v = vt[:, t - 1]

            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(torch.sigmoid(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v.T))

            vt_k = torch.zeros(n_v, gibbs_k, dtype=torch.float)
            ht_k = torch.zeros(n_h, gibbs_k, dtype=torch.float)
            for kk in range(gibbs_k):
                h = torch.bernoulli(
                    torch.sigmoid(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v.T))
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

            rt[:, t] = torch.sigmoid(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        return vt, rt

    def sample(self, v_start, chain=50, pre_gibbs_k=100, gibbs_k=20, mode=1, disable_tqdm=False):
        T = len(self.temporal_layers)
        n_h, n_v = self.W.shape

        vt = torch.zeros(n_v, chain + 1, dtype=torch.float)
        rt = torch.zeros(n_h, chain + 1, dtype=torch.float)
        rt[:, 0] = torch.sigmoid(torch.matmul(self.W, v_start.T) + self.b_init)
        vt[:, 0] = v_start
        for t in tqdm(range(1, chain + 1), disable=disable_tqdm):
            v = vt[:, t - 1]
            for kk in range(pre_gibbs_k):
                h = torch.bernoulli(
                    torch.sigmoid(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v.T))

            vt_k = torch.zeros(n_v, gibbs_k, dtype=torch.float)
            ht_k = torch.zeros(n_h, gibbs_k, dtype=torch.float)
            for kk in range(gibbs_k):
                h = torch.bernoulli(
                    torch.sigmoid(torch.matmul(self.W, v).T + self.b_h + torch.matmul(self.U, rt[:, t - 1]))).T
                v = torch.bernoulli(torch.sigmoid(torch.matmul(self.W.T, h) + self.b_v.T))
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

            rt[:, t] = torch.sigmoid(torch.matmul(self.W, vt[:, t]) + self.b_h + torch.matmul(self.U, rt[:, t - 1]))

        return vt[:, 1:], rt[:, 1:]


def learn(rtrbm, data, n_epochs, batch_size=10, lr=1e-3, CDk=10, enable_scheduler=False, momentum=0, dampening=0,
          weight_decay=0, shuffle_batch=True, plot=False, **kwargs):
    optimizer = torch.optim.SGD(rtrbm.parameters(), lr=lr, momentum=momentum, dampening=dampening,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True, factor=.5,
                                                           cooldown=10)

    if data.ndim == 3:
        n_batches = data.shape[2] // batch_size
    elif data.ndim == 2:
        n_batches = 1
        data = data[..., None]

    errors = []
    for epoch in tqdm(range(n_epochs)):
        err = 0
        for batch in range(n_batches):
            v_data = data[..., batch:batch + batch_size].to('cpu')
            with torch.no_grad():
                v_model, r_data, r_model = rtrbm.forward(v_data, CDk=CDk)

            optimizer.zero_grad()
            loss = rtrbm._parallel_pseudo_loglikelihood(v_data, r_data) - \
                   rtrbm._parallel_pseudo_loglikelihood(v_model, r_model)
            # loss = rtrbm._parallel_AIS_loglikelihood_AIS(v_data, r_data) - \
            #        rtrbm._parallel_AIS_loglikelihood_AIS(v_model, r_model)
            err += torch.sum((v_data - v_model) ** 2).detach().cpu()
            loss.backward()
            optimizer.step()

        errors += [err / data.numel()]
        if enable_scheduler:
            scheduler.step(errors[-1])
        if shuffle_batch:
            data[..., :] = data[..., torch.randperm(data.shape[2])]

    if plot:
        # Infer from trained RTRBM and plot some results
        vt_infer, rt_infer = rtrbm.infer(torch.tensor(data[:, :280 // 2, 0]))

        # effective coupling
        W = rtrbm.W.detach().numpy()
        U = rtrbm.U.detach().numpy()
        r_data = rtrbm._sample_r_given_v_over_time(data).detach().numpy()
        data = data.detach().numpy()
        var_h_matrix = np.reshape(np.var(r_data[..., 0], 1).repeat(W.shape[1]), [W.shape[1], W.shape[0]]).T
        var_v_matrix = np.reshape(np.var(data[..., 0], 1).repeat(W.shape[0]), [W.shape[0], W.shape[1]])

        Je_Wv = np.matmul(W.T, W * var_h_matrix) / W.shape[1] ** 2
        Je_Wh = np.matmul(W * var_v_matrix, W.T) / W.shape[0] ** 2

        _, ax = plt.subplots(2, 3, figsize=(12, 12))
        sns.heatmap(vt_infer[:, 270:].detach().numpy(), ax=ax[0, 0], cbar=False)
        ax[0, 0].set_title('Infered data')
        ax[0, 0].set_xlabel('Time')
        ax[0, 0].set_ylabel('Neuron index')

        ax[0, 1].plot(errors)
        ax[0, 1].set_title('RMSE of the RTRBM over epoch')
        ax[0, 1].set_xlabel('Epoch')
        ax[0, 1].set_ylabel('RMSE')

        sns.heatmap(Je_Wv, ax=ax[0, 2])
        ax[0, 2].set_title('Effective coupling V')
        ax[0, 2].set_xlabel("Visibel nodes")
        ax[0, 2].set_ylabel("Visibel nodes")

        sns.heatmap(W, ax=ax[1, 0])
        ax[1, 0].set_title('Visible to hidden connection')
        ax[1, 0].set_xlabel('Visible')
        ax[1, 0].set_ylabel('Hiddens')

        sns.heatmap(U, ax=ax[1, 1])
        ax[1, 1].set_title('Hidden to hidden connection')
        ax[1, 1].set_xlabel('Hidden(t-1)')
        ax[1, 1].set_ylabel('Hiddens(t)')

        sns.heatmap(Je_Wh, ax=ax[1, 2])
        ax[1, 2].set_title('Effective coupling H')
        ax[1, 2].set_xlabel("Hidden nodes [t]")
        ax[1, 2].set_ylabel("Hidden nodes [t]")
        plt.show()
    return errors


if __name__ == '__main__':

    from data.mock_data import create_BB
    import seaborn as sns
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from data.reshape_data import reshape

    #torch.autograd.set_detect_anomaly(True)

    train = create_BB(N_V=16, T=320, n_samples=10, width_vec=[4, 5, 6], velocity_vec=[1, 2])
    n_h = 8
    n_v, T, n_batches = train.shape
    rtrbm = RTRBM(n_hidden=n_h, n_visible=n_v, T=T)
    errors = learn(rtrbm, data=train, batch_size=10, n_epochs=500, CDk=10, lr=1e-3, plot=True)




