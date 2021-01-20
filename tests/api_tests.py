import sys
import h5py
import numpy as np
sys.path.append('./')

from tests.utils.assert_that import AssertThat 
from tests.utils.printer import print_test_result 
from src.utils.config_loader import load_config
from src.api import get_loader, get_buffer, get_generator, get_model

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

def run_tests():
    config = load_config('./tests/test_config.json')
    test_API_get_loader(config)
    test_API_get_buffer(config)
    test_API_get_generator(config)

run_tests()