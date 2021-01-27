class RawDataGenerator():
    def __init__(self, raw_data_loader):
        self.loader = raw_data_loader

    def get_batched_read(self):
        while True:
            x, read_id = self.loader.get_read()
            yield x, read_id