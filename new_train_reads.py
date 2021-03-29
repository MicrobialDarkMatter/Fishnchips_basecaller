import os 
import numpy as np


signal_path = './data/train_data/signals'
label_path = './data/train_data/labels'

read_ids = []
for file in os.scandir(signal_path):
    read_ids.append(file.name.split('.')[0])

signal_filepath = f'{signal_path}/{read_ids[0]}.npy'
label_filepath = f'{label_path}/{read_ids[0]}.npy'
signal = np.load(signal_filepath)
label = np.load(label_filepath)

print(label.shape)
