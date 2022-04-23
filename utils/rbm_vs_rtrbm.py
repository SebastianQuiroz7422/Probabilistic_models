from boltzmann_machines.data.reshape_data import *
from boltzmann_machines.data import *
import time
from utils.plots import *
from boltzmann_machines.cp_rtrbm import RTRBM
from utils.moments_plot import infer_and_get_moments_plot
from boltzmann_machines.cp_rbm import RBM
from boltzmann_machines.data import get_split_data


def zebrafish_compare(
        path='../data/part brain',
        save_fig_path=None,
        N_V=1000, N_H=10,
        pre_gibbs_k=0, gibbs_k=100,
        train_batches=80, test_batches=20,
        return_machines=False,
        **kwargs
):

    # getting the zebrafish data
    train, test = get_split_data(N_V=N_V, train_batches=train_batches, test_batches=test_batches)

    # training the rbm and rtrbm
    if return_machines:
        rbm, rtrbm = train_rbm_rtrbm(train_data=train, N_H=N_H, save_path=path, return_machines=return_machines,
                                     debug_mode=True, device='cuda', **kwargs)
    else:
        train_rbm_rtrbm(train_data=train, N_H=N_H, save_path=path, return_machines=False, debug_mode=True,
                        device='cuda', **kwargs)

    # creating the plots
    ax, res_rbm, res_rtrbm = compare_moments_trained(path, test, save_path=save_fig_path, pre_gibbs_k=pre_gibbs_k,
                                                     gibbs_k=gibbs_k)

    if return_machines:
        return ax, res_rbm, res_rtrbm, rbm, rtrbm
    else:
        return ax, res_rbm, res_rtrbm


def compare_moments_trained(path, test_set, save_path=None, pre_gibbs_k=0, gibbs_k=100):
    fig, axes = plt.subplots(5, 2, figsize=(15, 30))

    print('inferring and calculating moments for RBM...')
    _, _, res_rtrbm = infer_and_get_moments_plot(path + '/rtrbm.pt', test_set, n=10000, machine='rtrbm',
                                                 ax=axes[:, 1], fig=fig, pre_gibbs_k=pre_gibbs_k, gibbs_k=gibbs_k)

    print('inferring and calculating moments for RTRBM...')
    _, _, res_rbm = infer_and_get_moments_plot(path + '/rbm.pt', test_set, n=10000, machine='rbm',
                                               ax=axes[:, 0], fig=fig, pre_gibbs_k=pre_gibbs_k, gibbs_k=gibbs_k)

    axes[0, 0].set_title('RBM', fontsize=15)
    axes[0, 1].set_title('RTRBM', fontsize=15)
    axes[0, 0].set_ylabel(r'$<v_i>$', fontsize=18)
    axes[1, 0].set_ylabel(r'$<h_i>$', fontsize=18)
    axes[2, 0].set_ylabel(r'$<v_i v_j> - <v_i><v_j>$', fontsize=18)
    axes[3, 0].set_ylabel(r'$<h_i h_j> - <h_i><h_j>$', fontsize=18)
    axes[4, 0].set_ylabel(r'$<v_i h_j> - <v_i><h_j>$', fontsize=18)
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    return axes, res_rbm, res_rtrbm


def train_rbm_rtrbm(
        train_data,
        N_H=10,
        save_path=None,
        return_machines=False,
        debug_mode=True,
        device='cuda',
        **kwargs
):

    """Trains one rbm and one rtrbm witch exactly the same learning parameters.

    Parameters
    ----------
    train_data : 3D torch.Tensor
        The training data for the rbm and rtrbm to learn on
    N_H : int, 10 by default
        The number of hidden units
    save_path : str
        If specified, saves machines to file location
    return_machines : bool
        if True, returns the trained rbm and rtrbm
    debug_mode : bool
        if True, saves parameter history during training in machine.parameter_history
    device : 'cuda'  or 'cpu'
        device on which the machine are trained

    Returns
    -------
    rbm model
        the trained rbm
    rtrbm model
        the trained rtrbm
    """

    # initialize and train rbm
    print('training RBM...')
    rbm = RBM(train_data, N_H=N_H, debug_mode=debug_mode, device=device)
    rbm.learn(**kwargs)

    # save rbm
    if save_path is not None:
        print('saving RBM...')
        torch.save(rbm, save_path + '/rbm.pt')
        time.sleep(2)

    # delete rbm for memory purpose
    if not return_machines:
        del rbm

    # initialize and train rtrbm
    print('training RTRBM...')
    rtrbm = RTRBM(train_data, N_H=N_H, debug_mode=debug_mode, device=device)
    rtrbm.learn(**kwargs)

    # save rtrbm
    if save_path is not None:
        print('saving RTRBM...')
        torch.save(rtrbm, save_path + '/rtrbm.pt')
        time.sleep(2)

    # delete rbm for memory purpose
    if not return_machines:
        del rtrbm

    # return rbm and rtrbm if return_machine is specified as True
    if return_machines:
        return rbm, rtrbm
    else:
        return


if __name__ == '__main__':
    ax, res_rbm, res_rtrbm = zebrafish_compare(path='../boltzmann_machines/data/part brain',
                                               save_fig_path='../figures/test.png',
                                               N_V=1000, N_H=10,
                                               pre_gibbs_k=0, gibbs_k=100,
                                               train_batches=80, test_batches=20,
                                               n_epochs=10, lr=1e-3)
    plt.show()
