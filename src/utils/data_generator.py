import numpy as np

class DataGenerator():
    def __init__(self, data_buffer, label_window_size):
        self.data_buffer = data_buffer
        self.label_window_size = label_window_size

    def get_batches(self, amount):
        while True:
            batches = []
            for _ in range(amount):
                x,y,y_lens = next(self.get_batch())
                batches.append([x,y,y_lens])
            yield batches
        
    def get_batch(self):
        while True:
            x, y_raw = self.data_buffer.get_batch()
            y, y_lens = self.convert_to_target_language(y_raw)
            yield (x,y,y_lens)

    def get_batched_read(self):
        while True:
            x, y_raw, ref, dacs, read_id = self.data_buffer.get_batched_read()
            y, _ = self.convert_to_target_language(y_raw)
            yield x, y, ref, dacs, read_id

    def convert_to_target_language(self, y_raw):
        y = []
        y_lens = []
        # start_token = 5
        # end_token = 6
        for y_window in y_raw:
            y_window = [b+1 for b in y_window] 
            # y_window.insert(0, start_token)
            # y_window.append(end_token)
            label_length = len(y_window)
            padding_len = self.label_window_size - label_length
            y_window.extend([0]*padding_len)
            y.append(y_window)
            y_lens.append(label_length)
        return np.array(y), np.array(y_lens)

