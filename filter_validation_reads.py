import h5py
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import DataLoader

def plot_signal(signal):
    std = round(np.std(signal), 2)
    mean = round(np.mean(signal), 2)
    plt.plot(signal)
    axes = plt.gca()
    axes.set_xlim([0,200000])
    axes.set_ylim([0,1000])
    print(std)
    print(mean)
    plt.show()

path = './data/training/Stenotrophomonas_pavanii_MSB1_4D/sloika_hdf5s/remapped_0000.hdf5'
with h5py.File(path, 'r') as h5file:
    signal = h5file['chunks'][0]
    label = h5file['labels'][0]

plot_signal(signal)



