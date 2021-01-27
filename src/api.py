from src.utils.data_loader import DataLoader
from src.utils.raw_data_loader import RawDataLoader
from src.utils.data_buffer import DataBuffer
from src.utils.data_generator import DataGenerator
from src.utils.raw_data_generator import RawDataGenerator
from src.controllers.model_controller import ModelController
from src.controllers.validation_controller import ValidationController
from src.controllers.training_controller import TrainingController
from src.controllers.testing_controller import TestingController
from src.controllers.file_controller import FileController

def get_loader(config, key='training'):
    data_path = config[key]['data']
    return DataLoader(data_path)

def get_raw_loader(config):
    data_path = config['testing']['data']
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

def get_raw_generator(config):
    raw_loader = get_raw_loader(config)
    return RawDataGenerator(raw_loader)

def get_model(config):
    model_controller = ModelController(config)
    model = model_controller.initialize_model()
    return model

def get_trained_model(config, experiment_name):
    model = get_model(config)
    trained_model_path = FileController(experiment_name).get_model_filepath()
    trained_model = model.load_weights(trained_model_path)
    return model

def setup_experiment(experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.create_experiment_dir()

def get_validation_controller(config):
    generator = get_generator(config, key='validation')
    return ValidationController(config, generator)

def get_training_controller(config, experiment_name, model):
    validation_controller = get_validation_controller(config)
    generator = get_generator(config, key='training')
    return TrainingController(config, experiment_name, model, generator, validation_controller)

def get_testing_controller(config, experiment_name, model, append=False):
    generator = get_raw_generator(config)
    return TestingController(config, experiment_name, model, generator, append)