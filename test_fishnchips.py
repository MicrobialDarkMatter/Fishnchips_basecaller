import tensorflow as tf
import src.api as api
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Config filepath.')
    parser.add_argument('-n', '--name', required=False, help='Name of the experiment. If none provided, name with a timestamp will be assigned.')
    parser.add_argument('-d', '--discard', required=False, default=False, action='store_true', help='Discard existing trained model (if one exists) and begin training from scratch.')
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

def main(config_path, experiment_name, discard_existing):
    config = api.get_config(config_path)
    api.setup_experiment(experiment_name)
    api.test(config, experiment_name, discard_existing)

if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.name, args.discard)
    print(' - Script has finished successfully.') 