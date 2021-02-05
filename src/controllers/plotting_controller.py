import numpy as np
import matplotlib.pyplot as plt

from src.controllers.file_controller import FileController

class PlottingController():
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.primary_color = 'crimson'
        self.secondary_color = 'limegreen'

    def save_validation_plot(self, data, path):
        plt.figure(figsize=(5,5))
        plt.plot(data, color=self.primary_color)
        plt.title = 'Validation'
        plt.subtitle = f'Experiment:{self.experiment_name}'
        plt.xlabel('epoch')
        plt.ylabel('validation loss (editdistance)')
        plt.savefig(path)
        plt.close()

    def save_training_plot(self, loss, acc, path):
        plt.figure(figsize=(5,5))
        plt.plot(loss, color=self.primary_color)
        plt.plot(acc, color=self.secondary_color)        
        plt.title = 'Training'
        plt.subtitle = f'Experiment:{self.experiment_name}'
        plt.xlabel('epoch')
        plt.ylabel('loss / accuracy')
        plt.savefig(path)
        plt.close()        

    def save_testing_plot(self, acc, path, title='all'):
        bins = [i for i in range(0,105,5)]
        _, ax = plt.subplots(figsize=(18,6))
        counts, bins, _ = ax.hist(acc, bins=bins, rwidth=0.4, color=self.primary_color)
        ax.set_xticks(bins)
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        for count, x in zip(counts, bin_centers):
            ax.annotate(str(int(count)), xy=(x, 0), xycoords=('data', 'axes fraction'), xytext=(0, -18), textcoords='offset points', va='top', ha='center')
        plt.title = f'Testing ({title})'
        plt.subtitle = f'Experiment:{self.experiment_name}'
        plt.xlabel('read identity')
        plt.ylabel('read count')
        plt.savefig(path)
        plt.close()     

    def save_learning_rate_plot(self, lr, path):
        plt.figure(figsize=(5,5))
        plt.plot(lr, color=self.primary_color)
        plt.title = 'Learning rate'
        plt.subtitle = f'Experiment:{self.experiment_name}'
        plt.xlabel('epoch')
        plt.ylabel('learning rate')
        plt.savefig(path)
        plt.close()   