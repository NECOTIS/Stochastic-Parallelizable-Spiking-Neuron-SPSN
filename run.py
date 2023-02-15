# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Anonymous
"""

import os
import json
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from datasets import heidelberg_dataset
from network import create_network, train, test


parser = argparse.ArgumentParser(description="SNN training")
parser.add_argument('--seed', type=int)
parser.add_argument('--dataset', type=str, default='heidelberg', choices=["heidelberg"])
parser.add_argument('--neuron', type=str, default='LIF', choices=["LIF", "SPSN-SB", "SPSN-GS", "Non-Spiking"])
parser.add_argument('--nb_epochs', type=int, default=200)
parser.add_argument('--tau_mem', type=float, default=2e-2, help='neuron membrane time constant')
parser.add_argument('--tau_syn', type=float, default=2e-2, help='neuron synaptic current time constant')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=128, help='nb of neurons in the hidden layer')
parser.add_argument('--nb_layers', type=int, default=3, help='nb of hidden layers')
parser.add_argument('--reg_thr', type=float, default=0., help='spiking frequency regularization threshold')
parser.add_argument('--loss_mode', type=str, default='mean', choices=["last", "max", "mean"])
parser.add_argument('--data_augmentation', type=str, default='False', choices=["True", "False"])
parser.add_argument('--h_shift', type=float, default=0.1, help='data augmentation random shift factor')
parser.add_argument('--scale', type=float, default=0.3, help='data augmentation random scale factor')
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--save_model', type=str, default='False', choices=["True", "False"])

args = parser.parse_args()
PARAMS = {
    "seed" : args.seed,
    "dataset" : args.dataset,
    "neuron" : args.neuron,
	"nb_epochs" : args.nb_epochs,
    "tau_mem" : args.tau_mem,
    "tau_syn" : args.tau_syn,
    "batch_size" : args.batch_size,
    "hidden_size" : args.hidden_size,
    "nb_layers" : args.nb_layers,
    "reg_thr" : args.reg_thr,
    "loss_mode" : args.loss_mode,
    "data_augmentation" : args.data_augmentation=='True',
	"h_shift" : args.h_shift,
    "scale" : args.scale,
    "dir" : args.dir,
	"save_model" : args.save_model=='True',
   }







def save_results(train_results, test_results, PARAMS, model):
    """
    This function creates a dictionary of results from the training and testing and save it
    as a json file. 
    If the 'save_model' parameter is set to True, the trained model is also saved. 
    """
    outputs = {
       'loss_hist':train_results['loss'], 
       'train_accuracies':train_results['acc'], 
       'train_duration':train_results['dur'], 
       'test_accuracies': test_results['acc'],
       'nb_spikes':test_results['spk'],
       'test_duration':test_results['dur'],
       'PARAMS': PARAMS
      }
    
    output_dir = f"outputs/{PARAMS['dir']}"
    timestamp = int(datetime.timestamp(datetime.now()))
    filename = output_dir+'results_{}_{}.json'.format(PARAMS['neuron'], str(timestamp))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(outputs, f)
    
    if PARAMS['save_model']: 
        modelname = output_dir+'model_{}_{}.pt'.format(PARAMS['neuron'], str(timestamp))
        torch.save(model.state_dict(), modelname)



def main():
    """
    This function :
        - Enable or not the reproductibility by setting a seed
        - Loads the train and test sets
        - Create the network
        - Train and test the network
        - Save the results
    """
    print("\n-- Start --\n")
    #To enable reproductibility
    if PARAMS["seed"] is not None:
        seed=PARAMS["seed"]
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loads the train and test sets
    if PARAMS["dataset"]=="heidelberg":
        (train_set, test_set, input_size, nb_class, collate_fn_train,  
         collate_fn_test, max_duration) = heidelberg_dataset(device=device, augment=PARAMS["data_augmentation"], 
                                                        h_shift=PARAMS["h_shift"], scale=PARAMS["scale"])
    PARAMS["input_size"]=input_size
    PARAMS["nb_class"]=nb_class
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=PARAMS['batch_size'], shuffle=True, collate_fn=collate_fn_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=PARAMS['batch_size'], shuffle=False, collate_fn=collate_fn_test)
    
    # Create the network
    model = create_network(PARAMS, device, max_duration)
    
    # Train and test the network
    print("\n-- Training --\n")
    train_results = train(model, train_loader, nb_epochs=PARAMS['nb_epochs'], loss_mode=PARAMS['loss_mode'], reg_thr=PARAMS['reg_thr'])
    print("\n-- Testing --\n")
    test_results = test(model, test_loader, loss_mode=PARAMS['loss_mode'])
    
    # Save train and test results
    save_results(train_results, test_results, PARAMS, model)
    print("\n-- End --\n")
    


if __name__ == "__main__":
    main()













