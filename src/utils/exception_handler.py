def print_exception(message, exception=None, trance=False):
    if trace:
        traceback.print_exc()
    if exception is not None:
        print(exception)
    print(message)