# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Arnaud Yarga
"""

import torch
import numpy as np



class Base(torch.nn.Module):
    """
    Base class for creating a spiking neural network using PyTorch.

    Parameters:
    - input_size (int): size of input tensor
    - hidden_size (int): size of hidden layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True)
    - tau_mem (float, optional): time constant for the membrane potential (default: 1e-3)
    - tau_syn (float, optional): time constant for the synaptic potential (default: 1e-3)
    - time_step (float, optional): step size for updating the LIF model (default: 1e-3)
    - debug (bool, optional): flag to turn on/off debugging mode (default: False)
    """
    def __init__(self, input_size, hidden_size, device,
                 fire, tau_mem, tau_syn, time_step, debug):
        super(Base, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.v_th = torch.tensor(1.0)
        self.fire = fire
        self.debug = debug
        self.nb_spike_per_neuron = torch.zeros(self.hidden_size, device=self.device)

        # Neuron time constants
        self.alpha = float(np.exp(-time_step/tau_syn))
        self.beta = float(np.exp(-time_step/tau_mem))
        self.beta_1 = 1-self.beta

        # Fully connected layer for synapses
        self.fc = torch.nn.Linear(self.input_size, self.hidden_size, device=self.device)

        # Initializing weights
        torch.nn.init.kaiming_uniform_(self.fc.weight, a=0, mode='fan_in', nonlinearity='linear')
        torch.nn.init.zeros_(self.fc.bias)
        if self.debug:
            torch.nn.init.ones_(self.fc.weight)


