import wandb
import yaml
import traceback 
import src.api as api
import src.data_api as data_api
import tensorflow as tf
from pprint import pprint
from src.utils.data_loader import DataLoader
from src.utils.data_buffer import DataBuffer
from src.utils.data_generator import DataGenerator
from src.model.Attention.CustomSchedule import CustomSchedule
from src.model.Attention.attention_utils import create_combined_mask

import os
import uuid
import argparse
from src.utils.config_loader import load_config, verify_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sweep', required=True, help='Weights and biases config defining hyperparameters to optimize, optmization method, etc.')
    parser.add_argument('-i', '--iterations', required=True, help='Number of sweep iterations.')

    args = parser.parse_args()
    verify_args(args)
    return args

def verify_args(args):
    assert os.path.exists(args.sweep), 'Sweep config filepath is not valid.'
    assert int(args.iterations) > 0 and int(args.iterations) <= 1000, 'Number of iterations must be between 1 and 1000.'

def train(config=None):
    try:
        with wandb.init(config=config):
            wandb_config = wandb.config
            fnch_config = load_config('./configs/sweep_claaudia.json')
            fnch_config = build_model_config_from_wandb(wandb_config, fnch_config)
            verify_config(fnch_config)

            name = f'hyperparameter_opt_{str(uuid.uuid4())}'
            _ = api.train(fnch_config, name, new_training=True)
    except Exception as e:
        print(10 * '=')
        print(traceback.print_exc())
        print(e)
        print(10 * '=')


def build_model_config_from_wandb(wandb_config, fnch_config):
    fnch_config['model']['cnn_blocks'] = wandb_config.cnn_blocks
    fnch_config['model']['maxpool_idx'] = wandb_config.cnn_blocks // 2
    fnch_config['model']['maxpool_kernel'] = wandb_config.maxpool_kernel
    fnch_config['model']['attention_blocks'] = wandb_config.attention_blocks
    fnch_config['model']['num_heads'] = wandb_config.num_heads
    fnch_config['model']['dff'] = wandb_config.dff
    fnch_config['model']['signal_window_size'] = wandb_config.signal_window_size
    fnch_config['model']['label_window_size'] = wandb_config.signal_window_size // 3
    fnch_config['model']['dropout_rate'] = wandb_config.dropout_rate
    fnch_config['training']['signal_window_stride'] = wandb_config.signal_window_size // 3

    return fnch_config

def main(sweep_config_path, iterations):
    wandb.login()
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_config, project="fishandchips_basecaller")
        wandb.agent(sweep_id, train, count=iterations)

if __name__ == "__main__":
    args = parse_args()
    main(args.sweep, args.iterations)
    print(' - Script has finished successfully.')