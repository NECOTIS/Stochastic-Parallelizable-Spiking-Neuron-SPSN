# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Anonymous
"""

import time
import torch
import numpy as np
from tqdm import tqdm
from neurons.lif import LIF
from neurons.spsn import SPSN
import torch.nn.functional as F


def create_network(params, device, max_duration):
    """
    This function creates a neural network based on the given parameters
    """
    neuron = params["neuron"]
    nb_layers = params["nb_layers"]
    input_size = params["input_size"]
    hidden_size = params["hidden_size"]
    nb_class = params["nb_class"]
    tau_mem = params["tau_mem"]
    tau_syn = params["tau_syn"]

    modules = []
    if neuron=="LIF":
        modules.append(LIF(input_size, hidden_size, device, tau_mem=tau_mem, tau_syn=tau_syn))
        for i in range(nb_layers-1):
            modules.append(LIF(hidden_size, hidden_size, device, tau_mem=tau_mem, tau_syn=tau_syn))
        modules.append(LIF(hidden_size, nb_class, device, tau_mem=tau_mem, tau_syn=tau_syn, fire=False))
        
    elif neuron in ["SPSN-SB", "SPSN-GS"]:
        spike_mode = neuron.split('-')[-1]
        modules.append(SPSN(input_size, hidden_size, device, spike_mode, max_duration, tau_mem=tau_mem, tau_syn=tau_syn))
        for i in range(nb_layers-1):
            modules.append(SPSN(hidden_size, hidden_size, device, spike_mode, max_duration, tau_mem=tau_mem, tau_syn=tau_syn))
        modules.append(SPSN(hidden_size, nb_class, device, spike_mode, max_duration, tau_mem=tau_mem, tau_syn=tau_syn, fire=False))
        
    elif neuron=="Non-Spiking":
        modules.append(torch.nn.Linear(input_size, hidden_size, device=device))
        modules.append(torch.nn.ReLU())
        for i in range(nb_layers-1):
            modules.append(torch.nn.Linear(hidden_size, hidden_size, device=device))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(hidden_size, nb_class, device=device))
    model = torch.nn.Sequential(*modules)

    return model


def train(model, data_loader, nb_epochs=100, loss_mode='mean', reg_thr=0.):
    """
    This function Train the given model on the train data.
    """
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # If a regularization threshold is set we compute the theta_reg*N parameter of Equation (21)
    if reg_thr>0: 
        reg_thr_sum = reg_thr * np.sum([layer.hidden_size for layer in model if (layer.__class__.__name__ in ['LIF', 'SPSN'] and layer.fire)])

    loss_hist = []
    acc_hist = []
    progress_bar = tqdm(range(nb_epochs), desc=f"Train {nb_epochs} epochs")
    start_time = time.time()
    # Loop over the number of epochs
    for i_epoch in progress_bar:
        local_loss = 0
        local_acc = 0
        total = 0
        nb_batch = len(data_loader)
        # Loop over the batches
        for i_batch,(x,y) in enumerate(data_loader):
            total += len(y)
            output = model(x)
            # Select the relevant function to process the output based on loss mode
            if loss_mode=='last' : output = output[:,-1,:]
            elif loss_mode=='max': output = torch.max(output,1)[0] 
            else: output = torch.mean(output,1)

            # Here we set up our regularizer loss as in Equation (21)
            reg_loss_val = 0
            if reg_thr>0:
                spks = torch.stack([layer.nb_spike_per_neuron.sum() for layer in model if (layer.__class__.__name__ in ['LIF', 'SPSN'] and layer.fire)])
                reg_loss_val = F.relu(spks.sum()-reg_thr_sum)**2

            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(output, y) + reg_loss_val

            # Backpropagation and weights update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            local_loss += loss_val.detach().cpu().item()
            _,y_pred = torch.max(output,1)
            local_acc += torch.sum((y==y_pred)).detach().cpu().numpy()
            progress_bar.set_postfix(loss=local_loss/total, accuracy=local_acc/total, _batch=f"{i_batch+1}/{nb_batch}")
        
        loss_hist.append(local_loss/total)
        acc_hist.append(local_acc/total)

    train_duration = (time.time()-start_time)/nb_epochs
    
    return {'loss':loss_hist, 'acc':acc_hist, 'dur':train_duration}


def test(model, data_loader, loss_mode='mean'):
    """
    This function Computes classification accuracy for the given model on the test data.
    """
    acc = 0
    total = 0
    spk_per_layer = []
    progress_bar = tqdm(data_loader, desc="Test")
    start_time = time.time()
    # loop through the test data
    for x,y in progress_bar:
        total += len(y)
        with torch.no_grad():
            output = model(x)
            # Select the relevant function to process the output based on loss mode
            if loss_mode=='last' : output = output[:,-1,:]
            elif loss_mode=='max': output = torch.max(output,1)[0] 
            else: output = torch.mean(output,1)
            # get the predicted label
            _,y_pred = torch.max(output,1)
            acc += torch.sum((y==y_pred)).cpu().numpy()
            # get the number of spikes per layer for LIF and SPSN layers
            spk_per_layer.append([layer.nb_spike_per_neuron.sum().cpu().item() for layer in model if (layer.__class__.__name__ in ['LIF', 'SPSN'] and layer.fire)])
            progress_bar.set_postfix(accuracy=acc/total)
    test_duration = (time.time()-start_time)
    
    return {'acc':acc/total, 'spk':np.mean(spk_per_layer,axis=0).tolist(), 'dur':test_duration}
















