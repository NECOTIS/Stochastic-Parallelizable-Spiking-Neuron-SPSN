# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Anonymous
"""

import torch
from neurons.base import Base


class SPSN(Base):
    """
    Class for implementing a Stochastic and Parallelizable Spiking Neuron (SPSN) model

    Parameters:
    - input_size (int): size of input tensor
    - hidden_size (int): size of hidden layer
    - device (torch.device): device to use for tensor computations, such as 'cpu' or 'cuda'
    - spike_mode (str): "SB" for Sigmoid-Bernoulli or "GS" for Gumbel-Softmax
    - nb_steps (int): number of timesteps of inputs
    - fire (bool, optional): flag to determine if the neurons should fire spikes or not (default: True)
    - tau_mem (float, optional): time constant for the membrane potential (default: 1e-3)
    - tau_syn (float, optional): time constant for the synaptic potential (default: 1e-3)
    - time_step (float, optional): step size for updating the LIF model (default: 1e-3)
    - debug (bool, optional): flag to turn on/off debugging mode (default: False)
    """
	
    def __init__(self, input_size, hidden_size, device, spike_mode, nb_steps=None,
                 fire=True, tau_mem=1e-3, tau_syn=1e-3, time_step=1e-3, debug=False):

        super(SPSN, self).__init__(input_size, hidden_size, device,
                 fire, tau_mem, tau_syn, time_step, 
                 debug)
        # Set the spiking function
        if spike_mode=="SB": self.spike_fn = SigmoidBernoulli(self.device)
        elif spike_mode=="GS": self.spike_fn = GumbelSoftmax(self.device)

        self.nb_steps = nb_steps
        # Parameters can be computed upstream if the number of timesteps "nb_steps" is known
        self.fft_l_k = self.compute_params_fft(self.nb_steps)


    def compute_params_fft(self, nb_steps):
        """
        Compute the FFT of the parameters for parallel Leaky Integration

        Returns:
        fft_l_k: Product of FFT of parameters l and k
        """
        if nb_steps is None: return None

        l = torch.pow(self.alpha,torch.arange(nb_steps,device=self.device))
        k = torch.pow(self.beta,torch.arange(nb_steps,device=self.device))*self.beta_1
        fft_l = torch.fft.rfft(l, n=2*nb_steps).unsqueeze(1)
        fft_k = torch.fft.rfft(k, n=2*nb_steps).unsqueeze(1)

        return fft_l*fft_k


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

        # Recompute FFT params if nb_steps has changed
        if self.nb_steps!=nb_steps: 
            self.fft_l_k = self.compute_params_fft(nb_steps)
            self.nb_steps = nb_steps

        # Perform parallel leaky integration
        fft_X = torch.fft.rfft(X, n=2*nb_steps, dim=1)
        mem_rec = torch.fft.irfft(fft_X*self.fft_l_k, n=2*nb_steps, dim=1)[:,:nb_steps:,] # Equation (15)

        if self.fire:
        	# Perform stochastic firing
            spk_rec = self.spike_fn(mem_rec)
            self.nb_spike_per_neuron = torch.mean(torch.mean(spk_rec,dim=0),dim=0)
            return (spk_rec, mem_rec) if self.debug else spk_rec
        return mem_rec




class StochasticStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.bernoulli(input) # Equation (18)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input*input # Equation (19)



class SigmoidBernoulli(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.spike_fn = StochasticStraightThrough.apply
    
    def forward(self, inputs):
        spk_prob = torch.sigmoid(inputs) # Equation (17)
        spk = self.spike_fn(spk_prob)
        return spk






class GumbelSoftmax(torch.nn.Module):
    def __init__(self, device, hard=True, tau=1.0):
        super().__init__()
        
        self.hard = hard
        self.tau = tau
        self.uniform = torch.distributions.Uniform(torch.tensor(0.0).to(device),
                                                   torch.tensor(1.0).to(device))
        self.softmax = torch.nn.Softmax(dim=0)
  
    
    def forward(self, logits):
        # Sample uniform noise
        unif = self.uniform.sample(logits.shape + (2,))
        # Compute Gumbel noise from the uniform noise
        gumbels = -torch.log(-torch.log(unif))
        # Apply softmax function to the logits and Gumbel noise
        y_soft = self.softmax(torch.stack([(logits + gumbels[...,0]) / self.tau,
                                                     (-logits + gumbels[...,1]) / self.tau]))[0]
        if self.hard:
            # Use straight-through estimator
            y_hard = torch.where(y_soft > 0.5, 1.0, 0.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Use reparameterization trick
            ret = y_soft
            
        return ret