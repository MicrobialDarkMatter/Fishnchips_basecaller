import h5py
import matplotlib.pyplot as plt

# with h5py.File('./data/validation_catcaller.hdf5', 'r') as h5file:
#     keys = list(h5file.keys())
#     for key in keys:
#         read = h5file[key]
#         signal = read['Dacs'][:]
#         label = read['Reference'][:]
#         for i,_ in enumerate(signal):
#             sw = signal[i]
#             lw = label[i]
#             plt.plot(sw)
#             # print(signal[i][-50:])
#             print(lw)
#             if i == 0:
#                 break
#         break

from src.utils.raw_data_loader import RawDataLoader
# loader = RawDataLoader('./data/filtered_test_data/Klebsiella_pneumoniae_INF032', 300, 300)
# x, read_id = loader.get_read()
# plt.plot(x[0])
# plt.plot(x[1])
# plt.plot(x[2])
# plt.plot(x[3])
# plt.plot(x[4])
# axes = plt.gca()
# axes.set_xlim([0,300])
# axes.set_ylim([-10,10])
# plt.show()

import numpy as np
from src.utils.test_data_loader import TestDataLoader

loader = TestDataLoader('./data/cat_test_data')
signal, read_id = loader.get_read()
print(read_id)
for i,window in enumerate(signal):
    if i == 1:
        break
    plt.plot(window)

axes = plt.gca()   
axes.set_xlim([0,300])
axes.set_ylim([-10,10])
plt.show()