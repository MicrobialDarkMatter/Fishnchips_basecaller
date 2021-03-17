import h5py
import numpy as np
from collections import deque

class DataLoader():
    def __init__(self, data_filepath):
        self.data_filepath = data_filepath

    def load_read_ids(self):
        with h5py.File(self.data_filepath, 'r') as h5file:
            return list(h5file.keys())

    def load_read(self, read_id, normilize=False):
        with h5py.File(self.data_filepath, 'r') as h5file:
            read = h5file[read_id]
            dac = read['Dacs'][:]
            ref = read['Reference'][:]
            if normilize:
                dac = self.normalize(dac)
        return dac, ref

    def normalize(self, signal):
        signal = np.array(signal)
        mean = np.mean(signal)
        standard_dev = np.std(signal)
        return (signal - mean)/standard_dev