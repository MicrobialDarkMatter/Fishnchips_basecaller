import tensorflow as tf
import argparse
import os

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.utils.config_loader import load_config
from src.controllers.ui_controller import UI_Controller
from src.controllers.file_controller import File_Controller
from src.controllers.model_controller import Model_Controller
from src.controllers.training_controller import Training_Controller

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config filepath.')
    parser.add_argument('-n', '--name', required=False, help='Name of the experiment. If none provided, name with a timestamp will be assigned.')
    args = parser.parse_args()
    verify_args(args)
    return args

def verify_args(args):
    assert os.path.exists(args.config), 'Config filepath is not valid.'
    assert type(args.name) == str, 'Experiment name must be a string.'

def get_user_input(ui_controller):
    ui_controller.print_parameters('model')
    ui_controller.ask_parameters()
    ui_controller.print_parameters('training')
    ui_controller.print_parameters('validation')
    ui_controller.print_parameters('testing')
    ui_controller.ask_parameters()
    ui_controller.ask_retrain()
    ui_controller.ask_retest()
    return ui_controller.retrain, ui_controller.retest

def main(config_path, experiment_name):
    
    # TODO: Load model vs retrain
    config = load_config(config_path)
    ui_controller = UI_Controller(config, experiment_name)
    io_controller = File_Controller(experiment_name)
    model_controller = Model_Controller(config)

    retrain, retest = get_user_input(ui_controller)
    io_controller.create_experiment_dir()  
    model = model_controller.initialize_model()

    training_controller = Training_Controller(config, experiment_name, model, retrain)
    training_controller.train()

    

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.name)
    
    
     