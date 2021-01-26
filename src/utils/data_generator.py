import numpy as np

class DataGenerator():
    def __init__(self, data_buffer):
        self.data_buffer = data_buffer
        self.batch_count = 0

    def get_batches(self, amount):
        while True:
            batches = []
            for _ in range(amount):
                x,y = next(self.get_batch())
                batches.append([x,y])
            yield batches
        
    def get_batch(self):
        while True:
            self.batch_count += 1
            x, y_raw = self.data_buffer.get_batch()
            y = self.convert_to_target_language(y_raw)
            yield (x,y)

    def get_batched_read(self):
        while True:
            self.batch_count += 1
            x, y_raw, ref, dacs, read_id = self.data_buffer.get_batched_read()
            y = self.convert_to_target_language(y_raw)
            yield x, y, ref, dacs, read_id

    def convert_to_target_language(self, y_raw):
        y = []
        start_token = 5
        end_token = 6
        for y_window in y_raw:
            y_window = [b+1 for b in y_window] 
            y_window.insert(0, start_token)
            y_window.append(end_token)
            y.append(y_window)
        return np.array(y)

