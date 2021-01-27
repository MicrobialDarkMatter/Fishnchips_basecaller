import os
import json

def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    verify_config(config)
    return config

def verify_config(config):
    verify_model_config(config['model'])
    verify_train_config(config['training'], config['model'])
    vefify_validation_config(config['validation'], config['model'])
    verify_test_config(config['testing'], config['model'])
    print(' - Config was successfully loaded.')

def verify_model_config(model_config):
    assert type(model_config['signal_window_size']) == int and model_config['signal_window_size'] > 0, 'Max length of Encoder must be a positive integer. It corresponds to input signal window size.'
    assert type(model_config['label_window_size']) == int and model_config['label_window_size'] > 0, 'Max length of Decoder must be a positive integer. It corresponds to output label window size.'
    assert type(model_config['attention_blocks']) == int and model_config['attention_blocks'] > 0, 'Number of attention blocks must be a positive integer.'
    assert type(model_config['cnn_blocks']) == int and model_config['cnn_blocks'] > 0, 'Number of CNN blocks must be a positive integer.'
    assert type(model_config['maxpool_idx']) == int and model_config['maxpool_idx'] >= 0 & model_config['maxpool_idx'] < model_config['cnn_blocks'], 'Max pool idx must be a non-negative integer. It is intended to determine between which CNN block a maxpool layer is placed.'
    assert type(model_config['d_model']) == int and model_config['d_model'] > 0, 'Depth of the model must be a positive integer.'
    assert type(model_config['dff']) == int and model_config['dff'] > 0, 'Depth of the feed forward network must be a posotove integer.'
    assert type(model_config['num_heads']) == int and model_config['num_heads'] > 0, 'Number of attention heads must be a positive integer.'
    assert type(model_config['dropout_rate']) == float and 0 <= model_config['dropout_rate'] < 1, 'Drop out rate must be a float between 0 (included) and 1.'
    assert type(model_config['maxpool_kernel']) == int and model_config['label_window_size'] % model_config['maxpool_kernel'] == 0, 'Maxpool kernel size must be a positive integer and must devide window length with a remainder of 0.'

def verify_train_config(train_config, model_config):
    assert os.path.exists(train_config['data']), 'Invalid train data directory.'
    assert type(train_config['epochs']) == int and train_config['epochs'] > 0, 'Number of training epochs must be a positive integer.'
    assert type(train_config['warmup']) == int and train_config['warmup'] >= 0, 'Number of warmup epochs must be a positive integer.'
    assert type(train_config['patience']) == int and train_config['patience'] >= 0, 'Training patience (the number of epochs to wait for accuracy improvement) must be a positive integer.'
    assert type(train_config['batches']) == int and train_config['batches'] > 0, 'Number of training batches of an epoch must be a positive integer.'
    assert type(train_config['batch_size']) == int and train_config['batch_size'] > 0, 'Training batch size must be a positive integer.'
    assert type(train_config['buffer_size']) == int and train_config['buffer_size'] > 0, 'Training buffer size must be a positive integer.'
    assert type(train_config['lr_mult']) == int or type(train_config['lr_mult']) == float, 'Learning rate multiplier must be an int or a foat.'
    assert type(train_config['signal_window_stride']) == int and train_config['signal_window_stride'] > 0 and train_config['signal_window_stride'] <= model_config['signal_window_size'], 'Training signal window stride must be a non-negative integer, smaller or equal to signal windows size (otherwise signal values are skipped).'

def vefify_validation_config(validation_config, model_config):
    assert os.path.exists(validation_config['data']), 'Invalid validation data path.'
    assert type(validation_config['batch_size']) == int and validation_config['batch_size'] >= 0, 'Validation batch size must be a positive integer.'
    assert type(validation_config['buffer_size']) == int and validation_config['buffer_size'] > 0, 'Validation buffer size must be a positive integer.'
    assert type(validation_config['reads']) == int and validation_config['reads'] > 0, 'Number of reads to validate must be a positive integer.'
    assert type(validation_config['signal_window_stride']) == int and validation_config['signal_window_stride'] > 0 and validation_config['signal_window_stride'] <= model_config['signal_window_size'], 'Validation signal window stride must be a non-negative integer, smaller or equal to signal windows size (otherwise signal values are skipped).'

def verify_test_config(test_config, model_config):
    assert os.path.exists(test_config['data']), 'Invalid test data directory.'
    assert os.path.exists(test_config['reference']), 'Invalid test reference filepath.'
    assert type(test_config['batch_size']) == int and test_config['batch_size'] >= 0, 'Test batch size must be a positive integer.'
    assert type(test_config['buffer_size']) == int and test_config['buffer_size'] > 0, 'Test buffer size must be a positive integer.'
    assert type(test_config['signal_window_stride']) == int and test_config['signal_window_stride'] > 0 and test_config['signal_window_stride'] <= model_config['signal_window_size'], 'Test signal window stride must be a non-negative integer, smaller or equal to signal windows size (otherwise signal values are skipped).'
    assert type(test_config['reads']) == int and test_config['reads'] > 0, 'Number of reads to test must be a positive integer.'