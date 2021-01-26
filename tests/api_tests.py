
import os
import sys
import h5py
import numpy as np
import tensorflow as tf
sys.path.append('./')

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tests.utils.assert_that import AssertThat 
from tests.utils.printer import print_test_result 
from src.utils.config_loader import load_config
from src.api import get_loader, get_buffer, get_generator, get_model, get_trained_model
from src.model.FishNChips import FishNChips

def test_API_get_loader(config):
    test_name = test_API_get_loader.__name__
    expected = 5
    
    loader = get_loader(config, key='validation')
    reads_ids = loader.load_read_ids()
    actual = len(reads_ids)

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_API_get_buffer(config):
    test_name = test_API_get_buffer.__name__
    expected = 5
    
    buffer = get_buffer(config, key='validation')
    actual = len(buffer.read_ids)
    
    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_API_get_generator(config):
    test_name = test_API_get_generator.__name__
    expected = 10
    
    generator = get_generator(config, key='validation')
    batches = next(generator.get_batches(10))
    actual = len(batches)
    
    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_API_get_model(config):
    test_name = test_API_get_model.__name__
    actual = get_model(config)
    result = AssertThat(actual).is_instance_of(FishNChips)
    print_test_result(result, test_name, 'Is isntance of FishNChips', actual)

def test_API_get_trained_model(config, experiment_name):
    test_name = test_API_get_trained_model.__name__
    actual = get_trained_model(config, experiment_name)
    result = AssertThat(actual).is_instance_of(FishNChips)
    print_test_result(result, f'{test_name}', 'Is isntance of FishNChips', actual)

def run_tests():
    config = load_config('./configs/test_config.json')
    experiment_name = 'test_experiment'
    test_API_get_loader(config)
    test_API_get_buffer(config)
    test_API_get_generator(config)
    test_API_get_model(config)
    test_API_get_trained_model(config, experiment_name)

run_tests()