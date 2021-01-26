import h5py
import numpy as np
from collections import deque

class DataLoader():
    def __init__(self, data_filepath):
        self.data_filepath = data_filepath

    def load_read_ids(self):
        with h5py.File(self.data_filepath, 'r') as h5file:
            return list(h5file.keys())

    def load_read(self, read_id):
        with h5py.File(self.data_filepath, 'r') as h5file:
            read = h5file[read_id]
            dac = read['Dacs'][:]
            dac = self.normalize(dac)
            ref = deque(read['Reference'][:])
            rts = deque(read['Ref_to_signal'][:])
        return dac, ref, rts

    def normalize(self, signal):
        signal = np.array(signal)
        mean = np.mean(signal)
        standard_dev = np.std(signal)
        return (signal - mean)/standard_dev