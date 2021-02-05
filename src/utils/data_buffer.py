import math
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
        
        if len(self.label_windows) < self.batch_size:
            self.empty_buffer()
            self.fill_buffer()
            self.shuffle_buffer()

        x = np.array(self.signal_windows[:self.batch_size])
        y = np.array(self.label_windows[:self.batch_size])
        self.signal_windows = self.signal_windows[self.batch_size+1:]
        self.label_windows = self.label_windows[self.batch_size+1:] 
        
        return x,y

    def get_batched_read(self):
        read_id = self.read_ids[self.position]
        dacs, ref, _ = self.data_loader.load_read(read_id)
        read_x, read_y = self.get_segmented_read(read_id)
        self.set_position(increment=1)
        return np.array(read_x), np.array(read_y), list(ref), dacs, read_id
        
    def fill_buffer(self):
        found = 0
        while found < self.size:
            i = self.position + found
            read_x, read_y = self.get_segmented_read(self.read_ids[i]) 
            read_x, read_y = self.shuffle(read_x, read_y)
            idx = math.floor(len(x) / 10)
            self.signal_windows.extend(read_x[:idx])
            self.label_windows.extend(read_y[:idx])
            found += 1      
        self.set_position(increment=found)

    def empty_buffer(self):
        self.signal_windows = []
        self.label_windows = []
                                              
    def shuffle_buffer(self):
        x = np.array(self.signal_windows)
        y = np.array(self.label_windows)
        x_shuffled, y_shuffled = self.shuffle(x,y)
        
        self.signal_windows = x_shuffled.tolist()
        self.label_windows = y_shuffled.tolist()

    def shuffle(self, x, y):
        assert len(x) == len(y)
        c = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
        np.random.shuffle(c)
        x_shuffled = c[:, :x.size//len(x)].reshape(x.shape)
        y_shuffled = c[:, x.size//len(x):].reshape(y.shape)
        return x_shuffled, y_shuffled


    def get_segmented_read(self, read_id):
        x_read = []
        y_read = []       
        dac, ref, rts = self.data_loader.load_read(read_id)
        
        curdacs  = deque( [[x] for x in dac[rts[0]:rts[0]+self.signal_window_size-self.signal_window_stride]], self.signal_window_size )
        curdacts = rts[0]+ self.signal_window_size-self.signal_window_stride
        labels  = deque([])
        labelts = deque([])
        
        while rts[0] < curdacts:
            labels.append(ref.popleft())
            labelts.append(rts.popleft())

        while curdacts+self.signal_window_stride < rts[-1]-self.signal_window_size:
            curdacs.extend([[x] for x in dac[curdacts:curdacts+self.signal_window_stride]])
            curdacts += self.signal_window_stride

            while rts[0] < curdacts:
                labels.append(ref.popleft())
                labelts.append(rts.popleft())

            while len(labelts) > 0 and labelts[0] < curdacts - self.signal_window_size:
                labels.popleft()
                labelts.popleft()

            if len(labels) >= self.min_label_window_size:
                x_read.append(list(curdacs))
                y_read.append(list(labels))
            
        return (x_read,y_read)