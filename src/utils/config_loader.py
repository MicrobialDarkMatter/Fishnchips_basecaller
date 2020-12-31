import os
import json
from src.utils.validation_algorithm import VALIDATION_ALGORITHM

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    verify_config(config)
    return config

def verify_config(config):
    verify_model_config(config['model'])
    verify_train_config(config['training'])
    vefify_validation_config(config['validation'])
    verify_test_config(config['testing'])
    print(' - Config was successfully loaded.')

def verify_model_config(model_config):
    assert type(model_config['encoder_max_length']) == int and model_config['encoder_max_length'] > 0, 'Max length of Encoder must be a positive integer. It corresponds to input window size.'
    assert type(model_config['decoder_max_length']) == int and model_config['decoder_max_length'] > 0, 'Max length of Decoder must be a positive integer. It corresponds to output window size.'
    assert type(model_config['attention_blocks']) == int and model_config['attention_blocks'] > 0, 'Number of attention blocks must be a positive integer.'
    assert type(model_config['cnn_blocks']) == int and model_config['cnn_blocks'] > 0, 'Number of CNN blocks must be a positive integer.'
    assert type(model_config['maxpool_idx']) == int and model_config['maxpool_idx'] >= 0 & model_config['maxpool_idx'] < model_config['cnn_blocks'], 'Max pool idx must be a non-negative integer. It is intended to determine between which CNN block a maxpool layer is placed.'
    assert type(model_config['d_model']) == int and model_config['d_model'] > 0, 'Depth of the model must be a positive integer.'
    assert type(model_config['dff']) == int and model_config['dff'] > 0, 'Depth of the feed forward network must be a posotove integer.'
    assert type(model_config['num_heads']) == int and model_config['num_heads'] > 0, 'Number of attention heads must be a positive integer.'
    assert type(model_config['dropout_rate']) == float and 0 <= model_config['dropout_rate'] < 1, 'Drop out rate must be a float between 0 (included) and 1.'
    assert type(model_config['maxpool_kernel']) == int and model_config['encoder_max_length'] % model_config['maxpool_kernel'] == 0, 'Maxpool kernel size must be a positive integer and must devide window length with a remainder of 0.'

def verify_train_config(train_config):
    assert os.path.exists(train_config['data']), 'Invalid train data directory.'
    assert type(train_config['epochs']) == int and train_config['epochs'] > 0, 'Number of training epochs must be a positive integer.'
    assert type(train_config['warmup']) == int and train_config['warmup'] >= 0, 'Number of warmup epochs must be a positive integer.'
    assert type(train_config['patience']) == int and train_config['patience'] >= 0, 'Training patience (the number of epochs to wait for accuracy improvement) must be a positive integer.'
    assert type(train_config['batch_size']) == int and train_config['batch_size'] >= 0, 'Training batch size must be a positive integer.'
    assert type(train_config['lr_mult']) == int or type(train_config['lr_mult']) == float, 'Learning rate multiplier must be an int or a foat.'

def vefify_validation_config(validation_config):
    assert os.path.exists(validation_config['data']), 'Invalid validation data directory.'
    assert type(validation_config['batch_size']) == int and validation_config['batch_size'] >= 0, 'Validation batch size must be a positive integer.'
    assert validation_config['algorithm'] in VALIDATION_ALGORITHM.get_options(), f'Validation algorith options are {VALIDATION_ALGORITHM.get_options()}'

def verify_test_config(test_config):
    assert os.path.exists(test_config['data']), 'Invalid test data directory.'
    assert type(test_config['batch_size']) == int and test_config['batch_size'] >= 0, 'Test batch size must be a positive integer.'
