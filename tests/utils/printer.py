def print_test_result(result, test_name, expected, actual):
    if result:
        print(f' $ {test_name} Passed.')
        return
    print(f' ! {test_name} FAILED. Expected: {expected}, Actual: {actual}')
