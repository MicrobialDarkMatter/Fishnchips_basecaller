import numpy as np
import matplotlib.pyplot as plt

experiment_name = 'debug_clean'
a = np.load(f'trained_models/{experiment_name}/ctc.npy')

for example in a:
    for i in range(example.shape[-1]):
        base_time_steps = example[:,i]
        plt.plot(base_time_steps)
    plt.show()
    break