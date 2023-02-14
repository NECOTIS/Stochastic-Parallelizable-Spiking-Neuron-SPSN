# -*- coding: utf-8 -*-
"""
Created on February 2023

@author: Arnaud Yarga
"""
import os
import h5py
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T


# The Spiking Heidelberg Digits data set can be found at https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
def heidelberg_dataset(window_size=1, device=None, augment=False, h_shift=0., scale=0.):
    """
    Loads the Heidelberg dataset and sets up the training and testing datasets, feature and output sizes, and collate functions for data processing.

    Parameters:
    - window_size: size of the time window to be used as input to the model
    - device: device to use for PyTorch tensor computations (e.g. CPU or GPU)
    - augment: boolean flag indicating whether to perform data augmentation on the training set
    - h_shift: horizontal shift applied to the data during augmentation
    - scale: scale applied to the data during augmentation

    Returns:
    - train_set: PyTorch dataset for training
    - test_set: PyTorch dataset for testing
    - nb_features: number of features in the dataset
    - nb_class: number of outputs in the dataset
    - collate_fn_train: collate function for processing the training data
    - collate_fn_test: collate function for processing the testing data
    """

    nb_features = 700
    dt = 1e-3
    nb_class = 20
    train_set = Dataset_shd('datasets/shd_train.h5', dt, nb_features, window_size=window_size, device=device)
    test_set = Dataset_shd('datasets/shd_test.h5', dt, nb_features, window_size=window_size, device=device)
    max_duration = max(train_set.max_bin,test_set.max_bin)
    collate_fn_train = spikeTimeToMatrix_shd(device=device, augment=augment, h_shift=h_shift, scale=scale)
    collate_fn_test = spikeTimeToMatrix_shd(device=device, augment=False)

    return train_set, test_set, nb_features, nb_class, collate_fn_train, collate_fn_test, max_duration



class Dataset_shd(torch.utils.data.Dataset):
    """
    Custom torch Dataset for loading spike data from hdf5 file.

    Parameters:
    - path: path to the hdf5 file
    - dt: time step for spike data
    - nb_features: number of features for spike data
    - window_size: size of window for spike data (default = 1)
    - device: device to use for PyTorch tensor computations (e.g. CPU or GPU)
    """
    def __init__(self, path, dt, nb_features, window_size=1, device=None):
        super(Dataset_shd, self).__init__()
        assert os.path.exists(path), f"shd dataset not found at '{path}'. It is available at https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/"
        
        self.nb_features = nb_features
        self.device = device
        
        f = h5py.File(path, 'r')
        spikes_times_ = [k for k in f['spikes']['times']]
        spikes_units = [k.astype(np.int32) for k in f['spikes']['units']]
        self.labels = [k for k in f['labels']]
        f.close()
        
        # Get the maximum duration of the spikes data
        self.max_duration = int(max([t.max() for t in spikes_times_])/dt)
        self.max_bin = int(self.max_duration/window_size)+1
        bins = np.linspace(0, self.max_duration, num=self.max_bin)
        spikes_times_digitized = [np.digitize(t/dt,bins,right=True)  for t in spikes_times_]
        # Convert the digitized spike times and units to sparse tensors
        self.inputs_data = [self.to_sparse_tensor(spikes_t, spikes_u) for (spikes_t, spikes_u) in zip(spikes_times_digitized,spikes_units)]
    
    def to_sparse_tensor(self, spikes_times, spikes_units):
        """
        Convert digitized spike times and units to a sparse tensor
        spikes_times: digitized spike times
        spikes_units: units of the spikes
        """
        v = torch.ones(len(spikes_times))
        shape = [spikes_times.max()+1, self.nb_features]
        t = torch.sparse_coo_tensor(torch.tensor([spikes_times.tolist(), spikes_units.tolist()]), v, shape, dtype=torch.float32, device=self.device)
        return t
        
    def __getitem__(self, index):
        return self.inputs_data[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)



class spikeTimeToMatrix_shd(torch.nn.Module):
    """
    collate function for processing data. 
    Apply data augmentation and pad spike trains if necessary
    """

    def __init__(self, device=None, augment=False, h_shift=0.1, scale=0.3):
        super().__init__()
        self.device = device
        self.augment = augment
        self.h_shift = h_shift # channels axis
        self.scale = scale
        self.affine_transfomer = T.RandomAffine(degrees=0, translate=(self.h_shift, 0.), scale=(1-self.scale,1+self.scale))
    
    def forward(self, samples):
        max_d = max([st[0].shape[0] for st in samples])
        spike_train_batch = []
        labels_batch = []

        for (spike_train, label) in samples:
            # apply data augmentation if augment is True
            if self.augment:
                spike_train = self.affine_transfomer(spike_train.to_dense().unsqueeze(0)).squeeze(0)
            else:
                spike_train = spike_train.to_dense()

            # pad spike trains if needed
            pad = (0, 0, max_d-spike_train.shape[0], 0)
            spike_train_batch.append(F.pad(spike_train, pad, "constant", 0))
            labels_batch.append(label)

        return torch.stack(spike_train_batch), torch.tensor(labels_batch, device=self.device, dtype=torch.long)
    


        