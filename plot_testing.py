import src.api as api
import src.data_api as data_api
import matplotlib.pyplot as plt
from src.controllers.inference_controller import InferenceController
from src.utils.base_converter import convert_to_base_strings

experiment_name = 'experiment_claaudia_3'
config_path = './configs/test_config.json'

config = api.get_config(config_path)
model = api.get_trained_model(config, experiment_name)
batch_size = config['testing']['batch_size']

inference_controller = InferenceController()
for e in config['testing']['bacteria']:
    test_generator = data_api.get_raw_generator(config, e['data'])
    train_generator = data_api.get_generator(config, key='training')

    x_train, _ = next(test_generator.get_batched_read())
    x_test, _, _, _, _, = next(train_generator.get_batched_read())
    
    print(x_train.shape)
    print(x_test.shape)

    x_train = x_train[:100]
    x_test = x_test[:100]
    
    plt.ion()
    for i,_ in enumerate(x_train):
        xi_train = x_train[i]
        xi_test = x_test[i]

        plt.plot(xi_train)
        plt.plot(xi_test)

        plt.show()
        plt.pause(2)
        plt.close()
    
