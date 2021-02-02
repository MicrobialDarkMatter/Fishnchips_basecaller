import mappy as mp

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
from src.utils.config_loader import load_config

def get_config(path):
    return load_config(path)

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
    return RawDataGenerator(raw_loader)

def get_new_model(config):
    model_controller = ModelController(config)
    model = model_controller.initialize_model()
    return model

def get_trained_model(config, experiment_name):
    model = get_new_model(config)
    file_controller = FileController(experiment_name)
    assert file_controller.trained_model_exists(), ' ! Unable to load trained model. Invalid experiment name.'
    trained_model_path = file_controller.get_model_filepath()
    trained_model = model.load_weights(trained_model_path)
    return model

def setup_experiment(experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.create_experiment_dir()
    file_controller.create_assembly_directory()
    file_controller.create_report_directory()

def discard_existing_training(experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.teardown_training()
    file_controller.teardown_model()    

def discard_existing_testing(experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.teardown_evaluation()
    file_controller.teardown_assemblies()   
    
def get_validation_controller(config):
    generator = get_generator(config, key='validation')
    return ValidationController(config, generator)

def get_training_controller(config, experiment_name, model, discard_existing=False):
    validation_controller = get_validation_controller(config)
    generator = get_generator(config, key='training')
    return TrainingController(config, experiment_name, model, generator, validation_controller, discard_existing)

def train(config, experiment_name, new_training=False):
    if new_training:
        discard_existing_training(experiment_name)
        model = get_new_model(config)
    else:
        model = get_trained_model(config, experiment_name)
    controller = get_training_controller(config, experiment_name, model, new_training)
    trained_model = controller.train()
    return trained_model

def validate(config, experiment_name):
    model = get_trained_model(config, experiment_name)
    controller = get_validation_controller(config)
    editdistance = controller.validate(model)
    return editdistance

def test(config, experiment_name, new_testing=False):
    if new_testing:
        discard_existing_testing(experiment_name)
    model = get_trained_model(config, experiment_name)
    controller = TestingController(config, experiment_name, model, new_testing)

    for bacteria in config['testing']['bacteria']:
        name = bacteria['name']
        generator = get_raw_generator(bacteria['data'])
        aligner = mp.Aligner(bacteria['reference'])
        controller.test(name, generator, aligner)