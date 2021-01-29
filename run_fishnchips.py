import tensorflow as tf
import argparse
import os

from src.utils.config_loader import load_config
from src.controllers.ui_controller import UIController
from src.api import get_model, get_trained_model, setup_experiment, get_training_controller, get_testing_controller

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config filepath.')
    parser.add_argument('-n', '--name', required=False, help='Name of the experiment. If none provided, name with a timestamp will be assigned.')
    parser.add_argument('-w', '--warnings', required=False, default=False, action='store_true', help='Display tensorflow warning.')
    args = parser.parse_args()
    verify_args(args)
    set_logging(args)
    return args

def verify_args(args):
    assert os.path.exists(args.config), 'Config filepath is not valid.'
    assert type(args.name) == str, 'Experiment name must be a string.'

def set_logging(args):
    if args.warnings == False:
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_user_input(ui_controller):
    ui_controller.print_parameters('model')
    ui_controller.ask_parameters()
    ui_controller.print_parameters('training')
    ui_controller.print_parameters('validation')
    ui_controller.print_parameters('testing')
    ui_controller.ask_parameters()
    ui_controller.ask_retrain()
    ui_controller.ask_retest()
    return ui_controller.retrain, ui_controller.retest, ui_controller.append_test

def main(config_path, experiment_name):
    
    config = load_config(config_path)
    setup_experiment(experiment_name)

    ui_controller = UIController(config, experiment_name)
    retrain, retest, append_test = get_user_input(ui_controller)
    
    # TODO: Skip training, Start training, Continue training
    if retrain: 
        model = get_model(config)
        training_controller = get_training_controller(config, experiment_name, model)
        training_controller.train()
    else: 
        model = get_trained_model(config, experiment_name)
    
    if retest:
        testing_controller = get_testing_controller(config, experiment_name, model, append_test)
        testing_controller.test()

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.name)
    print(' - Script has finished successfully.')
    
    
     