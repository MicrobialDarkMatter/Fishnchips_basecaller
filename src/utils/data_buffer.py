import numpy as np

from collections import deque 
from src.utils.data_loader import DataLoader

class DataBuffer():
    def __init__(self, data_loader, buffer_size, batch_size, signal_window_size, signal_window_stride, min_labels_per_window=1):
        self.data_loader = data_loader
        self.position = 0
        self.size = buffer_size
        self.batch_size = batch_size
        self.signal_window_size = signal_window_size
        self.signal_window_stride = signal_window_stride
        self.min_label_window_size = min_labels_per_window
        self.signal_windows = []
        self.label_windows = []
        self.set_read_ids()      

    def set_read_ids(self):
        read_ids = self.data_loader.load_read_ids()
        np.random.shuffle(read_ids)
        self.read_ids = read_ids

    def set_position(self, increment):
        self.position = self.position + increment
        if self.position >= len(self.read_ids):
            self.position = 0
            self.set_read_ids()

    def get_batch(self):
        while len(self.label_windows) < self.batch_size:
            self.fetch()
            self.shuffle()

        x = np.array(self.signal_windows[:self.batch_size])
        y = np.array(self.label_windows[:self.batch_size])
        self.drop()
        return x,y

    def get_batched_read(self):
        read_id = self.read_ids[self.position]
        read_x, read_y = self.get_segmented_read(read_id)
        self.set_position(increment=1)
        return np.array(read_x), np.array(read_y), read_id
        
    def fetch(self):        
        skips, found = 0, 0
        while found < self.size:
            i = self.position + skips + found
            read_x, read_y = self.get_segmented_read(self.read_ids[i]) 

            self.signal_windows.extend(read_x)
            self.label_windows.extend(read_y)
            found += 1      
        self.set_position(increment=skips + found)  

    def drop(self):
        self.signal_windows = self.signal_windows[self.batch_size+1:]
        self.label_windows = self.label_windows[self.batch_size+1:]                                                 

    def shuffle(self):
        x = np.array(self.signal_windows)
        y = np.array(self.label_windows)

        c = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
        np.random.shuffle(c)
        x_shuffled = c[:, :x.size//len(x)].reshape(x.shape)
        y_shuffled = c[:, x.size//len(x):].reshape(y.shape)
        
        self.signal_windows = x_shuffled.tolist()
        self.label_windows = y_shuffled.tolist()

    def get_segmented_read(self, read_id):     
        x, y = self.data_loader.load_read(read_id)
        x = x.reshape((x.shape[0], x.shape[1], 1))
        return (x,y)