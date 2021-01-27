import os
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file

class RawDataLoader():
    def __init__(self, data_directory_path, signal_window_size, signal_window_stride):
        assert os.path.isdir(data_directory_path), f'Raw data loader\'s data_directory_path must point to an existing directory. Path: {data_directory_path}.'
        self.data_directory_path = data_directory_path       
        self.signal_window_size = signal_window_size
        self.signal_window_stride = signal_window_stride
        self.read_files = self.get_read_files()
        self.position = 0 

    def get_read_files(self):
        filenames = []
        for file in os.listdir(self.data_directory_path):
            if file.endswith('.fast5'):
                filenames.append(file)
        return filenames

    def get_read(self):
        read_filepath = f'{self.data_directory_path}/{self.read_files[self.position]}'
        self.position += 1
        dacs, read_id = self.load_file(read_filepath)
        dacs = self.normilize(dacs) 
        windows = self.segment_read(dacs)
        x = np.array(windows)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        return x, read_id
     
    def load_file(self, filepath):
        with get_fast5_file(filepath, mode="r") as f5:
            read_ids = f5.get_read_ids()
            assert len(read_ids) == 1, f'File {filepath} contains multiple reads.'    
            for read in f5.get_reads():
                return read.get_raw_data(), read.read_id
    
    def normilize(self, dacs):
        dacs = np.array(dacs)
        mean = np.mean(dacs)
        standard_dev = np.std(dacs)
        return (dacs - mean)/standard_dev

    def segment_read(self, dacs):
        windows = []
        while len(dacs) > self.signal_window_size:
            window = dacs[:self.signal_window_size]
            windows.append(window)
            dacs = dacs[self.signal_window_stride:]
        return windows