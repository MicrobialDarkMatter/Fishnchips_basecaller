import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import math
import src.evaluation_api as eval_api
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=False, help='Name of the experiment. If none provided, name with a timestamp will be assigned.')
    args = parser.parse_args()
    assert type(args.name) == str, 'Experiment name must be a string.'    
    return args  

def main(experiment_name):
    eval_api.plot_validation(experiment_name)
    eval_api.plot_training(experiment_name)
    eval_api.plot_testing(experiment_name)
    eval_api.plot_testing_per_bacteria(experiment_name)
    eval_api.make_report(experiment_name)

if __name__ == "__main__":
    args = parse_args()
    main(args.name)
    print(' - Script has finished successfully.')