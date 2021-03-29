import os
import h5py 
import numpy as np

target_path = './data/training_catcaller_300_signal.hdf5'
signal_path = '/Users/aau/Desktop/signal/out_signal_300'
label_path = '/Users/aau/Desktop/label/out_label_300'

read_ids = []
for file in os.scandir(signal_path):
    # if os.path.exists(f'{label_path}/{file}'):
    read_ids.append(file.name.split('.')[0])

with h5py.File(target_path, 'w') as f:
    for i,read_id in enumerate(read_ids):
        if i > 9:
            break
        assert read_id not in f, ' ! Duplicate read id. File is incorrectly batched.'
        signal_filepath = f'{signal_path}/{read_id}.npy'
        # label_filepath = f'{label_path}/{read_id}.npy'
        signal = np.load(signal_filepath)
        # label = np.load(label_filepath)
        g = f.create_group(read_id)
        g.create_dataset('Dacs', data=signal)
        # g.create_dataset('Reference', data=label)