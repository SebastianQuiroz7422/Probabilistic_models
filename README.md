Probabilistic models

Different versions of Boltzmann Machines (BM), the restricted BM (RBM) and the Recurrent Temporal RBM (RTRBM).
The RTRBM is also implemented with the autograd function of pytorch, however due to the intractability of the model the gradients are approximated by the psuedo log-likelihood, similiarly for the RTRBM with a gaussian hidden unit potential.

