from src.utils.data_loader import DataLoader
from src.utils.data_buffer import DataBuffer
from src.utils.data_generator import DataGenerator
from src.controllers.model_controller import ModelController
from src.controllers.validation_controller import ValidationController
from src.controllers.training_controller import TrainingController
from src.controllers.file_controller import FileController

def get_loader(config, key='training'):
    data_path = config[key]['data']
    return DataLoader(data_path)

def get_buffer(config, key='training'):
    loader = get_loader(config, key)
    buffer_size = config[key]['buffer_size']
    batch_size = config[key]['batch_size']
    signal_window_stride = config[key]['signal_window_stride']
    signal_window_size = config['model']['signal_window_size']
    return DataBuffer(loader, buffer_size, batch_size, signal_window_size, signal_window_stride)

def get_generator(config, key='training'):
    buffer = get_buffer(config, key)
    return DataGenerator(buffer)

def get_model(config):
    model_controller = ModelController(config)
    model = model_controller.initialize_model()
    return model

def get_validation_controller(config):
    generator = get_generator(config, key='validation')
    return ValidationController(config['validation'], generator)

def get_training_controller(config, experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.create_experiment_dir()

    validation_controller = get_validation_controller(config)
    model = get_model(config)
    generator = get_generator(config, key='training')
    return Training_Controller(config, experiment_name, model, generator, validation_controller)



