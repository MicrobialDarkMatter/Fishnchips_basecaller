import sys
import h5py
import numpy as np
sys.path.append('./')

from src.utils.data_buffer import DataBuffer
from tests.utils.assert_that import AssertThat  
from tests.utils.printer import print_test_result

def test_buffer_get_batch_x_shape(data_filepath):
    test_name = test_buffer_get_batch_x_shape.__name__
    expected = (32,300,1)
    
    batch_size = 32
    buffer_size = 5
    signal_window_size = 300
    signal_window_stride = 30
    min_labels_per_window = 1
    buffer = DataBuffer(data_filepath, buffer_size, batch_size, signal_window_size, signal_window_stride)
    x, _ = buffer.get_batch()
    actual = x.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_buffer_get_batch_y_shape(data_filepath):
    test_name = test_buffer_get_batch_y_shape.__name__
    expected = (32,)
    
    batch_size = 32
    buffer_size = 5
    signal_window_size = 300
    signal_window_stride = 30
    min_labels_per_window = 1
    buffer = DataBuffer(data_filepath, buffer_size, batch_size, signal_window_size, signal_window_stride)
    _, y = buffer.get_batch()
    actual = y.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def run_tests():
    data_filepath = './tests/data.hdf5'
    test_buffer_get_batch_x_shape(data_filepath)
    test_buffer_get_batch_y_shape(data_filepath)

run_tests()