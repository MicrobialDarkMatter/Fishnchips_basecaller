import sys
import h5py
import numpy as np
sys.path.append('./')

from src.utils.data_buffer import DataBuffer
from src.utils.data_loader import DataLoader
from tests.utils.assert_that import AssertThat  
from tests.utils.printer import print_test_result
from src.utils.data_generator import DataGenerator

def test_generator_get_batch_x_shape(buffer):
    setup(buffer)
    test_name = test_generator_get_batch_x_shape.__name__
    expected = (buffer.batch_size, buffer.signal_window_size, 1) #(32, 300, 1)
    
    generator = DataGenerator(buffer)
    x, _ = next(generator.get_batch())
    actual = x.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def test_generator_get_batch_y_shape(buffer):
    setup(buffer)
    test_name = test_generator_get_batch_y_shape.__name__
    expected = (buffer.batch_size,) #(32,)
    
    generator = DataGenerator(buffer)
    _, y = next(generator.get_batch())
    actual = y.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def test_generator_get_batch_y_start_token(buffer):
    setup(buffer)
    test_name = test_generator_get_batch_y_start_token.__name__
    expected = 5
    
    generator = DataGenerator(buffer)
    _, y = next(generator.get_batch())
    actual = y[0][0]

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def test_generator_get_batch_y_end_token(buffer):
    setup(buffer)
    test_name = test_generator_get_batch_y_end_token.__name__
    expected = 6
    
    generator = DataGenerator(buffer)
    _, y = next(generator.get_batch())
    actual = y[0][-1]

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def test_generator_get_batched_read_x_shape(buffer):
    setup(buffer)
    test_name = test_generator_get_batched_read_x_shape.__name__
    expected = (buffer.signal_window_size, 1) # (any, 300, 1)
    
    generator = DataGenerator(buffer)
    x, _, _, _, _ = next(generator.get_batched_read())
    actual = x.shape[1:] 

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def setup(buffer):
    buffer.position = 0

def teardown(buffer):
    buffer.position = 0

def run_tests():

    batch_size = 32
    buffer_size = 5
    signal_window_size = 300
    signal_window_stride = 30
    min_labels_per_window = 1
    loader = DataLoader('./tests/test_data.hdf5')
    buffer = DataBuffer(loader, buffer_size, batch_size, signal_window_size, signal_window_stride)

    test_generator_get_batch_x_shape(buffer)
    test_generator_get_batch_y_shape(buffer)
    test_generator_get_batch_y_start_token(buffer)
    test_generator_get_batch_y_end_token(buffer)
    test_generator_get_batched_read_x_shape(buffer)

run_tests()