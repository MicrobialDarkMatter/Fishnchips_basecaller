import sys
import h5py
import numpy as np
sys.path.append('./')

from src.utils.data_buffer import DataBuffer
from src.utils.data_loader import DataLoader
from tests.utils.assert_that import AssertThat  
from tests.utils.printer import print_test_result
from src.utils.data_generator import DataGenerator

def test_generator_get_batch_x_shape(buffer, label_window_size):
    setup(buffer)
    test_name = test_generator_get_batch_x_shape.__name__
    expected = (buffer.batch_size, buffer.signal_window_size, 1) #(32, 300, 1)
    
    generator = DataGenerator(buffer, label_window_size)
    x, _ = next(generator.get_batch())
    actual = x.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def test_generator_get_batch_y_shape(buffer, label_window_size):
    setup(buffer)
    test_name = test_generator_get_batch_y_shape.__name__
    expected = (buffer.batch_size, label_window_size) #(32,100)
    
    generator = DataGenerator(buffer, label_window_size)
    _, y = next(generator.get_batch())
    actual = y.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def test_generator_get_batch_y_start_token(buffer, label_window_size):
    setup(buffer)
    test_name = test_generator_get_batch_y_start_token.__name__
    expected = 5
    
    generator = DataGenerator(buffer, label_window_size)
    _, y = next(generator.get_batch())
    actual = y[0][0]

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)
    teardown(buffer)

def test_generator_get_batch_y_end_token(buffer, label_window_size):
    setup(buffer)
    test_name = test_generator_get_batch_y_end_token.__name__
    
    generator = DataGenerator(buffer, label_window_size)
    _, y = next(generator.get_batch())
    end_token_indexes = np.where(y[0] == 6)[0]

    result = AssertThat(len(end_token_indexes), 1).are_equal()
    print_test_result(result, test_name, 1, len(end_token_indexes))
    teardown(buffer)

def test_generator_get_batch_y_padding(buffer, label_window_size):
    setup(buffer)
    test_name = test_generator_get_batch_y_padding.__name__

    generator = DataGenerator(buffer, label_window_size)
    _, y = next(generator.get_batch())
    end_token_indexes = np.where(y[0] == 6)[0]
    i = end_token_indexes[0]

    padding = y[0][i+1:]
    zeros = np.where(padding == 0)[0]

    result = AssertThat(len(zeros), len(padding)).are_equal()
    print_test_result(result, f'{test_name}_only_zero_padding', len(padding), len(zeros))
    teardown(buffer)

def test_generator_get_batched_read_x_shape(buffer, label_window_size):
    setup(buffer)
    test_name = test_generator_get_batched_read_x_shape.__name__
    expected = (buffer.signal_window_size, 1) # (any, 300, 1)
    
    generator = DataGenerator(buffer, label_window_size)
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
    label_window_size = 100
    loader = DataLoader('./data/test_data.hdf5')
    buffer = DataBuffer(loader, buffer_size, batch_size, signal_window_size, signal_window_stride)

    test_generator_get_batch_x_shape(buffer, label_window_size)
    test_generator_get_batch_y_shape(buffer, label_window_size)
    test_generator_get_batch_y_start_token(buffer, label_window_size)
    test_generator_get_batch_y_end_token(buffer, label_window_size)
    test_generator_get_batch_y_padding(buffer, label_window_size)
    test_generator_get_batched_read_x_shape(buffer, label_window_size)

run_tests()