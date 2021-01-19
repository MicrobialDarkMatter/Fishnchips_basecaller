import sys
import h5py
import numpy as np
sys.path.append('./')

from src.utils.data_loader import DataLoader
from tests.utils.assert_that import AssertThat 
from tests.utils.printer import print_test_result  

def test_read_id_list_length(data_filepath):
    test_name = test_read_id_list_length.__name__
    expected = 5
    
    loader = DataLoader(data_filepath)
    reads = loader.load_read_ids()
    actual = len(reads)
    
    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_read_id_length(data_filepath):
    test_name = test_read_id_length.__name__
    expected = len('002f0f2d-ffc1-4072-82d3-6ce425d9724e')
    
    loader = DataLoader(data_filepath)
    actual = len(loader.load_read_ids()[0])

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_read_dacs_list_elemnt(data_filepath):
    test_name = test_read_dacs_list_elemnt.__name__
    expected = np.float64

    loader = DataLoader(data_filepath)
    read_id = loader.load_read_ids()[0]
    dacs, _, _ = loader.load_read(read_id)
    actual = type(dacs[0])

    result = AssertThat(actual, expected).are_equal()
    print_test_result(result, test_name, expected, actual)

def test_read_dacs_is_normilized(data_filepath):
    test_name = test_read_dacs_is_normilized.__name__
    
    loader = DataLoader(data_filepath)
    read_id = loader.load_read_ids()[0]
    dacs, _, _ = loader.load_read(read_id)
    arr = np.array(dacs)
    actual = np.std(arr)
    result = AssertThat(actual).is_in_interval(0.99, 1.01)
    print_test_result(result, test_name, "std of signal to be < 1.3 and > 0.7.", actual)

def run_tests():
    data_filepath = './tests/data.hdf5'
    test_read_id_list_length(data_filepath)
    test_read_id_length(data_filepath)
    test_read_dacs_list_elemnt(data_filepath)
    test_read_dacs_is_normilized(data_filepath)

run_tests()

