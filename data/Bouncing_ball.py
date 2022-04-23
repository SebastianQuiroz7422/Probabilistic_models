import torch
import numpy as np
import random

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