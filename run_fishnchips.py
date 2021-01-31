import tensorflow as tf
import src.api as api
import argparse
import os

from src.controllers.ui_controller import UIController

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
        print(' - Hiding tensorflow output messages.')
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
    return ui_controller

def main(config_path, experiment_name):
    config = api.get_config(config_path)
    api.setup_experiment(experiment_name)

    ui_controller = UIController(config, experiment_name)
    ui_controller = get_user_input(ui_controller)

    if ui_controller.skip_training == False:
        api.train(config, experiment_name, ui_controller.new_training)
    
    if ui_controller.skip_testing == False:
        api.test(config, experiment_name, ui_controller.new_testing)

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.name)
    print(' - Script has finished successfully.')
    
    
     