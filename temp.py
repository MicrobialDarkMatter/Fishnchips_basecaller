import h5py
import src.api as api
import src.data_api as data_api
import matplotlib.pyplot as plt
from src.utils.config_loader import load_config
from src.utils.base_converter import convert_to_base_strings

config = load_config('./configs/catcaller.json')
generator = data_api.get_generator(config)
raw_generator = data_api.get_raw_generator(config, "./data/filtered_test_data/Haemophilus_haemolyticus_M1C132_1")


path = '/user/cs.aau.dk/oh22ue/Preprocessing/out/training_catcaller_0.hdf5'
# path = './data/training_catcaller.hdf5'
with h5py.File(path, 'r') as h5file:
    keys = list(h5file.keys())

for i,read_id in enumerate(keys):
    with h5py.File(path, 'r') as h5file:
        read = h5file[read_id]
        dac = read['Dacs'][:]
        ref = read['Reference'][:]
        print(f'{type(ref)} |  {ref.shape} | {i}')
        assert ref.shape[-1] == 300, f'{ref.shape}'

# while True:
#     batches = next(generator.get_batches(100))
#     for i,(x,y) in enumerate(batches):
#         print(f'{y.shape} || {i}')        


# batch = next(generator.get_batch())
# train_xs = batch[0]
# train_ys = batch[1]
# train_ys = convert_to_base_strings(train_ys)

# test_xs, read_id = next(raw_generator.get_batched_read())
# print(test_xs.shape)
# print(train_xs.shape)

# for x,y in zip(xs,ys):
#     plt.plot(x)
#     plt.title(y)
#     axes = plt.gca()
#     axes.set_xlim([0,2500])
#     axes.set_ylim([-10,10])
#     plt.show()
#     plt.close()

# api.train(config, 'test_experiment_new_data',new_training=True)