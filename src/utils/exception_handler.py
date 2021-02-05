import traceback

def print_exception(message, exception=None, show_trance=False):
    if show_trance:
        traceback.print_exc()
    if exception is not None:
        print(exception)
    print(message)