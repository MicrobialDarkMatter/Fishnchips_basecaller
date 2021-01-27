import tensorflow as tf
import argparse
import os

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.utils.config_loader import load_config
from src.controllers.ui_controller import UIController
from src.api import get_model, get_trained_model, setup_experiment, get_training_controller

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
    
    config = load_config(config_path)
    setup_experiment(experiment_name)

    ui_controller = UIController(config, experiment_name)
    retrain, retest = get_user_input(ui_controller)

    # TODO: Load model vs retrain
    if retrain: 
        model = get_model(config)
        training_controller = get_training_controller(config, experiment_name, model)
        training_controller.train()

    if retest:
        pass

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.name)
    
    
     