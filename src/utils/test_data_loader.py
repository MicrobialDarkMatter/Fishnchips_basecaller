import os
import numpy as np

class TestDataLoader:
    def __init__(self, data_directory_path):
        self.data_directory_path = data_directory_path 
        self.read_files = self.get_read_files()
        self.position = 0

    def get_read_files(self):
        filenames = []
        for file in os.listdir(self.data_directory_path):
            if file.endswith('.npy'):
                filenames.append(file)
        print(len(filenames))
        return filenames
    
    def get_read(self):
        filename = self.read_files[self.position]
        filepath = f'{self.data_directory_path}/{filename}'
        self.position += 1
        x = np.load(filepath)
        read_id = filepath.split('/')[-1].split('.')[0]
        return x, read_id
