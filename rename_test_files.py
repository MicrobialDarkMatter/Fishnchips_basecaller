import os
from src.utils.raw_data_loader import RawDataLoader

data_dir = './data/all_test_data/'
read_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]  
for read_dir in read_dirs:
    loader = RawDataLoader(read_dir, None, None)
    files = loader.get_read_files()
    for file in files:
        try:
            filepath = f'{read_dir}/{file}'
            _, read_id = loader.load_file(filepath)
            new_filepath = f'{read_dir}/{read_id}.fast5'
            os.rename(filepath, new_filepath)
        except Exception as e:
            print(e)

# os.rename('./test.txt', 'lmao.txt')