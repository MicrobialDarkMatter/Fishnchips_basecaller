from src.utils.data_loader import DataLoader
from src.utils.raw_data_loader import RawDataLoader
from src.utils.data_buffer import DataBuffer
from src.utils.data_generator import DataGenerator
from src.utils.raw_data_generator import RawDataGenerator

def get_loader(config, key='training'):
    data_path = config[key]['data']
    return DataLoader(data_path)

def get_raw_loader(config, data_path):
    signal_window_size = config['model']['signal_window_size']
    signal_window_stride = config['testing']['signal_window_stride']
    return RawDataLoader(data_path, signal_window_size, signal_window_stride)

def get_buffer(config, key='training'):
    loader = get_loader(config, key)
    buffer_size = config[key]['buffer_size']
    batch_size = config[key]['batch_size']
    signal_window_stride = config[key]['signal_window_stride']
    signal_window_size = config['model']['signal_window_size']
    return DataBuffer(loader, buffer_size, batch_size, signal_window_size, signal_window_stride)

def get_generator(config, key='training'):
    buffer = get_buffer(config, key)
    label_window_size = config['model']['label_window_size']
    return DataGenerator(buffer, label_window_size)
  
def get_raw_generator(config, path):
    loader = get_raw_loader(config, path)
    return RawDataGenerator(loader)