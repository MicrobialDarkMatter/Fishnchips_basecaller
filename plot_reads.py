import os
import json
import matplotlib.pyplot as plt
from src.utils.raw_data_loader import RawDataLoader
from shutil import copyfile

for file in os.scandir('./data/filter'):
    bacteria = file.name.split('.')[0]
    path = f'./data/filter/{bacteria}.json'
    with open(path, 'r') as f:
        data = json.load(f)

    source_dir = f'./data/all_test_data/{bacteria}'
    target_dir = f'./data/filtered_test_data/{bacteria}'
    if os.path.exists(target_dir) == False:
        os.makedirs(target_dir)
    for i,read_id in enumerate(data['read_ids'][:20]):
        read_source_path = f'{source_dir}/{read_id}.fast5'
        read_target_path = f'{target_dir}/{read_id}.fast5'
        copyfile(read_source_path, read_target_path)
        
        # read_dict = loader.load_file(read_path)
        # signal, read_id = loader.scale(read_dict)

        # plt.plot(signal)
        # plt.plot(read_dict['raw'])
        # plt.title(f'{bacteria} || {i}')
        # axes = plt.gca()
        # axes.set_xlim([0,500000])
        # axes.set_ylim([0,2000])
        # plt.show(block=False)
        # plt.pause(0.01)
        # plt.clf()
        
