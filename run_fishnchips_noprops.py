import tensorflow as tf
import argparse
import os

# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.utils.config_loader import load_config
from src.api import get_model, get_trained_model, setup_experiment, get_training_controller, get_testing_controller

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

def main(config_path, experiment_name):
    
    config = load_config(config_path)
    setup_experiment(experiment_name)
    
    model = get_model(config)
    training_controller = get_training_controller(config, experiment_name, model)
    testing_controller = get_testing_controller(config, experiment_name, model)
    
    training_controller.train()
    testing_controller.test()

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.name)
    print(' - Script has finished successfully.')
    
    
     