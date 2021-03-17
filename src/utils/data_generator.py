import numpy as np

class DataGenerator():
    def __init__(self, data_buffer, label_window_size):
        self.data_buffer = data_buffer
        self.label_window_size = label_window_size

    def get_batches(self, amount):
        while True:
            batches = []
            for _ in range(amount):
                x,y = next(self.get_batch())
                batches.append([x,y])
            yield batches
        
    def get_batch(self):
        while True:
            x, y_raw = self.data_buffer.get_batch()
            y = self.convert_to_target_language(y_raw)
            yield (x,y)

    def get_batched_read(self):
        while True:
            x, y_raw, read_id = self.data_buffer.get_batched_read()
            y = self.convert_to_target_language(y_raw)
            yield x, y, read_id

    def convert_to_target_language(self, y_raw):
        y = []
        start_token = 6
        end_token = 5
        for y_window in y_raw:
            y_window = [int(b) for b in y_window if b < 5] # Remove padding
            y_window.insert(0, start_token) # Add start token
            y_window.append(end_token) # Add end token
            padding_len = self.label_window_size - len(y_window)
            y_window.extend([0]*padding_len) # Add padding (0s)
            y.append(y_window)
        return np.array(y)

