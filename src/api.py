import mappy as mp
import src.evaluation_api as evaluation_api
import src.data_api as data_api

from src.controllers.model_controller import ModelController
from src.controllers.validation_controller import ValidationController
from src.controllers.training_controller import TrainingController
from src.controllers.testing_controller import TestingController
from src.controllers.file_controller import FileController
from src.utils.config_loader import load_config

def get_config(path):
    return load_config(path)

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

def setup_experiment(experiment_name, config):
    file_controller = FileController(experiment_name)
    file_controller.create_experiment_dir()
    file_controller.create_assembly_directory()
    file_controller.create_report_directory()
    file_controller.create_prediction_directory()
    file_controller.save_config(config)

def discard_existing_training(experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.teardown_training()
    file_controller.teardown_model()    
    file_controller.teardown_ctc()

def discard_existing_testing(experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.teardown_testing()
    file_controller.teardown_assemblies()   
    file_controller.teardown_evaluation()

def discard_existing_evaluation(experiment_name):
    file_controller = FileController(experiment_name)
    file_controller.teardown_evaluation()

def get_validation_controller(config):
    generator = data_api.get_generator(config, key='validation')
    return ValidationController(config, generator)

def get_training_controller(config, experiment_name, model, discard_existing=False):
    validation_controller = get_validation_controller(config)
    generator = data_api.get_generator(config, key='training')
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
        generator = data_api.get_raw_generator(config, bacteria['data'])
        aligner = mp.Aligner(bacteria['reference'])
        controller.test(name, generator, aligner)

def evaluate(experiment_name):
    evaluation_api.plot_validation(experiment_name)
    evaluation_api.plot_training(experiment_name)
    evaluation_api.plot_testing(experiment_name)
    evaluation_api.plot_testing_per_bacteria(experiment_name)
    evaluation_api.plot_learning_rate(experiment_name)
    evaluation_api.make_report(experiment_name)