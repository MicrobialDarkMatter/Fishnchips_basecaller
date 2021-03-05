import os
import matplotlib.pyplot as plt
from ont_fast5_api.fast5_interface import get_fast5_file
from src.utils.raw_data_loader import RawDataLoader


def load_file(filepath):
        with get_fast5_file(filepath, mode="r") as f5:
            read_ids = f5.get_read_ids()
            assert len(read_ids) == 1, f'File {filepath} contains multiple reads.'    
            for read in f5.get_reads():
                channel_info = read.get_channel_info()
                return {
                    'read_id': read.read_id,
                    'raw':read.get_raw_data(),
                    'offset':channel_info['offset'],
                    'digitisation':channel_info['digitisation'],
                    'range':channel_info['range']
                }
path = './data/filtered_test_data/Haemophilus_haemolyticus_M1C132_1/5210_N128870_20170307_FN2002033683_MN19691_mux_scan_Klebs_Ecoli_HI_barcode_98141_ch68_read26_strand.fast5'
read = load_file(path)
print(read)


path = './data/filtered_test_data'
for directory in os.listdir(path):
    print(directory)
    if directory == '.DS_Store':
        continue
    loader = RawDataLoader(f'{path}/{directory}', 300, 300)
    loader.position = 1
    normilized, read_id = loader.get_read()
    plt.plot(normilized[1000:1300], label=directory)

# # plt.plot(scaled[:300], label='scaled', color='blue')
plt.legend()
plt.show()