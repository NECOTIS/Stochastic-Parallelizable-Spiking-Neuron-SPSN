# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Anonymous
"""

import torch
from neurons.base import Base


class LIF(Base):
    """
    Class for implementing a Leaky Integrate and Fire (LIF) neuron model

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
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, 
                 debug=False):

        super(LIF, self).__init__(input_size, hidden_size, device,
                 fire, tau_mem, tau_syn, time_step, 
                 debug)
        # Set the spiking function
        self.spike_fn = SurrGradSpike.apply


    def forward(self, inputs):
        """
        Perform forward pass of the network

        Parameters:
        - inputs (tensor): Input tensor with shape (batch_size, nb_steps, input_size)

        Returns:
        - Return membrane potential tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is False
        - Return spiking tensor with shape (batch_size, nb_steps, hidden_size) if 'fire' is True
        - Return the tuple (spiking tensor, membrane potential tensor) if 'debug' is True
        """
        X = self.fc(inputs)
        batch_size,nb_steps,_ = X.shape
        syn = torch.zeros((batch_size,self.hidden_size), device=self.device)
        mem = torch.zeros((batch_size,self.hidden_size), device=self.device)
        mem_rec = []
        spk_rec = []
        
        # Iterate over each time step
        for t in range(nb_steps):
            # Integrating input to synaptic current - Equation (5)
            syn = self.alpha*syn + X[:,t] 
            # Integrating synaptic current to membrane potential - Equation (6)
            mem = self.beta*mem + self.beta_1*syn 
            if self.fire:
                # Spikes generation - Equation (3)
                spk = self.spike_fn(mem-self.v_th) 
                spk_rec.append(spk)
                # Membrane potential reseting - Equation (6)
                mem = mem * (1-spk.detach()) 
            mem_rec.append(mem)
        
        mem_rec = torch.stack(mem_rec,dim=1)
        if self.fire:
            spk_rec = torch.stack(spk_rec,dim=1)
            self.nb_spike_per_neuron = torch.mean(torch.mean(spk_rec,dim=0),dim=0)
            return (spk_rec, mem_rec) if self.debug else spk_rec
        return mem_rec



# Surrogate gradient implementation from https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
class SurrGradSpike(torch.autograd.Function):
    scale = 100.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad