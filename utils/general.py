import numpy as np
import torch


class AttrDict(dict):
    """Class that acts like a dictionary + items can be accessed by attribute"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def add_dim(arr, dim):
    if isinstance(arr, torch.Tensor):
        arr = arr.unsqueeze(dim=dim)
    elif isinstance(arr, np.ndarray):
        arr = np.expand_dims(arr, axis=dim)
    else:
        raise Exception('Adding dimension is only supported for numpy arrays or torch tensors.')
    return arr


def concat(arrs, dim=0):
    if all(isinstance(arr, torch.Tensor) for arr in arrs):
        arrs = torch.cat(arrs, dim=dim)
    elif all(isinstance(arr, np.ndarray) for arr in arrs):
        arrs = np.concatenate(arrs, axis=dim)
    else:
        raise Exception('Concatenation is only supported for numpy arrays or torch tensors.')
    return arrs

