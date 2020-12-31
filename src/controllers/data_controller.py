import h5py
import tensorflow as tf

class DataController():
    def __init__(self, data_path, batch_size):
        self.data_path = data_path 
        self.batch_size = batch_size
        self.buffer_size = 2000

    def load_data(self, cap=None):
        with h5py.File(self.data_path, 'r') as f:
            x = f['x'][:cap]
            y = f['y'][:cap]
        assert x.shape[0] == y.shape[0], ' ! Data contains various number of examples and labels.'
        print(f' - - Data loaded successfully.')
        print(f' - - x | shape: {x.shape}, type: {x.dtype}')
        print(f' - - y | shape: {y.shape}, type: {y.dtype}')        
        return x,y
    
    def process_data(self, x, y):
        x = tf.constant(x, dtype=tf.float32)
        x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
        y = tf.constant(y, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((x,y))
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.buffer_size).padded_batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        print(' - - Data preprocessed successfully.')
        return dataset