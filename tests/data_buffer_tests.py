import sys
import h5py
import numpy as np
sys.path.append('./')

from src.utils.data_buffer import DataBuffer
from src.utils.data_loader import DataLoader
from tests.utils.assert_that import AssertThat  
from tests.utils.printer import print_test_result

def test_buffer_get_batch_x_shape(data_loader):
    test_name = test_buffer_get_batch_x_shape.__name__
    expected = (32,300,1)
    
    batch_size = 32
    buffer_size = 5
    signal_window_size = 300
    signal_window_stride = 30
    min_labels_per_window = 1
    buffer = DataBuffer(data_loader, buffer_size, batch_size, signal_window_size, signal_window_stride)
    x, _ = buffer.get_batch()
    actual = x.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_buffer_get_batch_y_shape(data_loader):
    test_name = test_buffer_get_batch_y_shape.__name__
    expected = (32,)
    
    batch_size = 32
    buffer_size = 5
    signal_window_size = 300
    signal_window_stride = 30
    min_labels_per_window = 1
    buffer = DataBuffer(data_loader, buffer_size, batch_size, signal_window_size, signal_window_stride)
    _, y = buffer.get_batch()
    actual = y.shape

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_buffer_get_batched_read_x_shape(data_loader):
    test_name = test_buffer_get_batched_read_x_shape.__name__
    
    batch_size = 32
    buffer_size = 5
    signal_window_size = 300
    signal_window_stride = 30
    min_labels_per_window = 1
    buffer = DataBuffer(data_loader, buffer_size, batch_size, signal_window_size, signal_window_stride)
    x, _, _, _, _ = buffer.get_batched_read()
    actual = np.array(x).shape

    result1 = AssertThat(actual[0]).is_in_interval(5e2, 1e5)
    result2 = AssertThat(actual[1], 300).are_equal()
    result3 = AssertThat(actual[2], 1).are_equal()

    print_test_result(result1, f'{test_name}_a', f'Value betweem {5e2} - {1e5}', actual[0])
    print_test_result(result1, f'{test_name}_b', 300, actual[1])
    print_test_result(result1, f'{test_name}_c', 1, actual[2])

def run_tests():
    data_loader = DataLoader('./tests/test_data.hdf5')
    test_buffer_get_batch_x_shape(data_loader)
    test_buffer_get_batch_y_shape(data_loader)
    test_buffer_get_batched_read_x_shape(data_loader)

run_tests()