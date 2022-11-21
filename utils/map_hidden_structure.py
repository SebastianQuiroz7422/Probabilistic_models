import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import torch
import numpy as np


def receptive_fields(weights, coordinates, only_max_conn=True):
    if only_max_conn:
        idx = torch.abs(weights) == torch.max(torch.abs(weights), 0)[0]
        return (torch.matmul(torch.abs(weights) * idx, coordinates).T / torch.sum(torch.abs(weights * idx), 1)).T
    else:
        return (torch.matmul(torch.abs(weights), coordinates).T / torch.sum(torch.abs(weights), 1)).T


class MapZebra(object):
    """This is a class to plot the neural assemblies of a trained (RT)RBM.
    It has a slider that is used to change the hidden unit and a plots for each dimension.
    To use, first create an instance of the class by giving the hidden-visible weights and the 3-dimensional coordinates
    of the neurons. Then call MapZebra.plot() to create the plot.
    It probably does not work in jupyter notebook.
    """
    def __init__(self, weights, coordinates, threshold=None):
        self.weights = np.array(weights)
        self.coordinates = np.array(coordinates)
        self.n_hidden, self.n_visible = self.weights.shape
        self.threshold = threshold

        # get min and max coordinate values
        self.x_min = np.min(self.coordinates[:, 0])
        self.x_max = np.max(self.coordinates[:, 0])
        self.y_min = np.min(self.coordinates[:, 1])
        self.y_max = np.max(self.coordinates[:, 1])
        self.z_min = np.min(self.coordinates[:, 2])
        self.z_max = np.max(self.coordinates[:, 2])

        # get the strongest connecting hidden unit
        self.strongest_connections = np.argmax(np.abs(weights), 0)

        # initialize plot
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))

    def update(self, hidden_unit):
        for ax in self.axes:
            ax.cla()
        hidden_unit = int(hidden_unit - 1)
        if hidden_unit == -1:
            self.axes[0].plot(self.coordinates[:, 0], self.coordinates[:, 1], '.', ms=1)
            self.axes[1].plot(self.coordinates[:, 1], self.coordinates[:, 2], '.', ms=1)
            self.axes[2].plot(self.coordinates[:, 0], self.coordinates[:, 2], '.', ms=1)
        else:
            visible_idx = self.strongest_connections == hidden_unit
            if self.threshold is not None:
                visible_idx *= np.abs(self.weights[hidden_unit, :]) > self.threshold
                visible_idx = torch.tensor(visible_idx, dtype=torch.bool)
            self.axes[0].plot(self.coordinates[visible_idx, 0], self.coordinates[visible_idx, 1], '.', ms=1)
            self.axes[1].plot(self.coordinates[visible_idx, 1], self.coordinates[visible_idx, 2], '.', ms=1)
            self.axes[2].plot(self.coordinates[visible_idx, 0], self.coordinates[visible_idx, 2], '.', ms=1)
        self.set_limits_labels()
        self.fig.canvas.draw_idle()

    def set_limits_labels(self):
        self.axes[0].set_xlim([self.x_min, self.x_max])
        self.axes[0].set_ylim([self.y_min, self.y_max])
        self.axes[1].set_xlim([self.y_min, self.y_max])
        self.axes[1].set_ylim([self.z_min, self.z_max])
        self.axes[2].set_xlim([self.x_min, self.x_max])
        self.axes[2].set_ylim([self.z_min, self.z_max])
        self.axes[0].set_xlabel('x')
        self.axes[0].set_ylabel('y')
        self.axes[1].set_xlabel('y')
        self.axes[1].set_ylabel('z')
        self.axes[2].set_xlabel('x')
        self.axes[2].set_ylabel('z')

    def plot_all(self):
        self.axes[0].plot(self.coordinates[:, 0], self.coordinates[:, 1], '.', ms=1)
        self.axes[1].plot(self.coordinates[:, 1], self.coordinates[:, 2], '.', ms=1)
        self.axes[2].plot(self.coordinates[:, 0], self.coordinates[:, 2], '.', ms=1)
        self.set_limits_labels()

        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Make a horizontal slider to control the hidden unit
        self.axSlider = plt.axes([0.25, 0.1, 0.65, 0.03])

        self.slider = Slider(
            ax=self.axSlider,
            label='#hidden unit',
            valmin=0,
            valmax=self.n_hidden + 1,
            valinit=0,
            valfmt='%d',
        )

        self.slider.on_changed(self.update)
        return self.fig.show()

    def plot_one(self, hidden_unit):
        visible_idx = self.strongest_connections == hidden_unit
        if self.threshold is not None:
            visible_idx *= np.abs(self.weights[hidden_unit, :]) > self.threshold
            visible_idx = visible_idx.type(torch.bool)
        self.axes[0].plot(self.coordinates[visible_idx, 0], self.coordinates[visible_idx, 1], '.', ms=1)
        self.axes[1].plot(self.coordinates[visible_idx, 1], self.coordinates[visible_idx, 2], '.', ms=1)
        self.axes[2].plot(self.coordinates[visible_idx, 0], self.coordinates[visible_idx, 2], '.', ms=1)
        self.set_limits_labels()
        return self.fig.show()

    def new_plot(self):
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        return


class MapHiddenStructure(object):
    def __init__(self, W=None, U=None, rtrbm=None, dir=None, coordinates=None):
        if W is not None:
            self.W = W
        if U is not None:
            self.U = U
        if dir is not None:
            rtrbm = torch.load(dir, map_location='cpu')
        if rtrbm is not None:
            self.W = rtrbm.W
            self.U = rtrbm.U
            if rtrbm.debug_mode:
                self.parameter_history = rtrbm.parameter_history

        self.cmap = plt.get_cmap('tab20')

        self.n_h, self.n_v = self.W.shape

        if coordinates is None:
            # assume an even distribution of visible neurons over populations
            coordinates = torch.empty(self.n_v, 2)
            self.n_v_pop = self.n_v // self.n_h
            theta = torch.linspace(0, 2 * torch.pi, self.n_h + 1)
            x, y = torch.cos(theta), torch.sin(theta)
            for pop in range(self.n_h):
                coordinates[pop * self.n_v_pop:(pop + 1) * self.n_v_pop, 0] = \
                    x[pop] + .2 * torch.sin(.5 * theta[1]) * torch.randn(self.n_v_pop)
                coordinates[pop * self.n_v_pop:(pop + 1) * self.n_v_pop, 1] = \
                    y[pop] + .2 * torch.sin(.5 * theta[1]) * torch.randn(self.n_v_pop)
        self.coordinates = coordinates

        self.rf = receptive_fields(self.W, self.coordinates)

    def draw_final_structure(self):
        ax = self.draw_structure(self.W, self.U)
        return ax

    def draw_structure_evolution(self, fig=None, ax=None, save=False, path=''):
        if fig is None and ax in None:
            fig, ax = plt.subplots()
        ani = FuncAnimation(fig, self.update, frames=len(self.parameter_history))
        return ax

    def update(self, frame):
        ax = self.draw_structure(self.parameter_history[frame][0], self.parameter_history[frame][0])
        return ax

    def draw_structure(self, W, U, hidden_weight_threshold=.5, r=.1, vr=2, save=False, path='', ax=None):
        if ax is None:
            ax = plt.subplot()

        theta = torch.deg2rad(torch.tensor(200))
        max_hidden_connection = torch.max(torch.abs(W), 0)[1]
        U_norm = U / torch.max(U)

        for h, (x, y) in enumerate(self.rf):
            # draw visible neurons as dots
            ax.scatter(self.coordinates[max_hidden_connection == h, 0],
                       self.coordinates[max_hidden_connection == h, 1],
                       color=self.cmap.colors[2 * h], s=vr)

            # draw hidden neurons as circles
            circle = plt.Circle((x, y), radius=r, fill=False, color=self.cmap.colors[2 * h])
            ax.add_patch(circle)
            ax.text(x, y - .03, str(h), ha='center', fontsize=10)

            # draw hidden connections as arrows
            for hh in range(self.n_h):
                u = U_norm[h, hh]
                color_ = 'red' if u < 0 else 'green'
                width_ = torch.abs(u)
                if abs(u) > hidden_weight_threshold:

                    # self-connecting arrow
                    if h == hh:
                        arc = Arc((x, y + r), r * 2, r * 2, angle=-30, theta1=0, theta2=230, capstyle='round',
                                  linestyle='-', lw=width_, color=color_)

                        X, Y = x + r * torch.cos(theta), y + r * torch.sin(theta)
                        arc_head = RegularPolygon((X, Y + r), 3, r / 5, theta.item(), color=color_)
                        ax.add_patch(arc_head)
                        ax.add_patch(arc)

                    # draw arrow
                    else:
                        x2, y2 = self.rf[hh]
                        angle = torch.atan2(x2 - x, y2 - y)
                        dx, dy = r * torch.sin(angle), r * torch.cos(angle)
                        ax.arrow(x+dx, y+dy, x2-x-2*dx, y2-y-2*dy, lw=width_, color=color_, length_includes_head=True,
                                 head_width=width_ / 30, overhang=0)
            ax.axis('off')
            ax.axis('square')

            if save:
                plt.savefig(path, dpi=500)
        return ax


if __name__ == '__main__':
    from data.load_data import get_split_data, load_data_thijs
    import sys, os

    sys.path.append(r'D:\OneDrive\RU\Intern\rtrbm_master\PGM\source')
    sys.path.append(r'D:\OneDrive\RU\Intern\rtrbm_master\PGM\utilities')
    import RBM_utils

    os.chdir(r'D:/OneDrive/RU\Intern/rtrbm_master/cRBM Jerome+Thijs/crbm_zebrafish_spontaneous_data')
    RBM = RBM_utils.loadRBM(
        'cRBM_models/RBM3_20180912-Run01-spontaneous-rbm2_wb_test-segs-267-nseg10_M200_l1-2e-02_duration208093s_timestamp2020-05-16-0844.data')

    _, coordinates, _ = load_data_thijs(
        data_path=r'neural_recordings\full_calcium_data_sets\20180912_Run01_spontaneous_rbm2.h5')

    spikes = np.load(
        r'neural_recordings\full_spike-only_data_sets\20180912_Run01_spontaneous_rbm2__wb_spikes_only.npy')

    spikes_idx = np.load(r'neural_recordings\full_spike-only_data_sets\spike_idx.npy')

    map = MapZebra(weights=RBM.weights, coordinates=coordinates[spikes_idx, :], threshold=.01)
    map.plot_all()
    plt.show()