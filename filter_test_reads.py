import os 
import json
import numpy as np
import matplotlib.pyplot as plt
from src.utils.raw_data_loader import RawDataLoader
from scipy.signal import find_peaks

NORMILIZE = False

def get_good_reads(path):
    loader = RawDataLoader(path, 300, 300)
    good_reads = []
    try:
        counter = 0
        while True:
            read, read_id = loader.get_read(normilize=NORMILIZE)
            print(f' - Loading read {counter+1}', end='\r')
            if is_good_read(read):
                good_reads.append(read_id)
            counter += 1
        return good_reads
    except Exception as e:
        print()
        print(e)
        return good_reads
    
def is_good_read(signal):
    signal = signal.reshape((signal.shape[0] * signal.shape[1]))
    inverted = signal * (-1)
    std = round(np.std(signal), 2)
    peaks, _ = find_peaks(signal, prominence=100)
    valleys, _ = find_peaks(inverted, prominence=100)
    noise = len(peaks) + len(valleys)
    plot_read(signal, peaks, valleys)
    return True if std < 14 and noise == 0 else False

def plot_read(signal, p, v):

    std = round(np.std(signal), 2)
    
    
    print(f'STD:{std}')
    plt.plot(signal)
    plt.plot(p, signal[p], "x")
    plt.plot(v, signal[v], "x")

    axes = plt.gca()
    axes.set_xlim([0,200000])
    axes.set_ylim([-20,20]) if NORMILIZE else axes.set_ylim([0,500])
    plt.show()

# data_dir = './data/all_test_data/'
# read_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]   
# for read_dir in read_dirs:
#     read_files = [f.path for f in os.scandir(read_dir)]
#     total_reads = len(read_files)
#     print(f'Scanning {read_dir} ({total_reads} reads)')
#     reads = get_good_reads(read_dir)
#     print(f'Found {len(reads)} good reads.')
#     save_path = read_dir.split('/')[-1]
#     with open(f'./data/filter/{save_path}.json', 'w') as f:
#         json.dump({
#             "total":total_reads,
#             "good":len(reads),
#             "read_ids":reads
#         }, f)


# get_good_reads('./data/randomly_selected_test_data/Haemophilus_haemolyticus_M1C132_1')
get_good_reads('./data/randomly_selected_test_data/Klebsiella_pneumoniae_INF042')