import numpy as np

from src.controllers.file_controller import FileController
from src.controllers.plotting_controller import PlottingController
from src.controllers.evaluation_controller import EvaluationController
from src.utils.exception_handler import print_exception

def plot_validation(experiment_name):
    try:
        file_controller = FileController(experiment_name)
        path = file_controller.get_validation_plot_filepath()
        data = file_controller.load_training()
        data = np.array(data)
        data = data[:,2]
        data = data[1:]
        plotting_controller = PlottingController(experiment_name)
        plotting_controller.save_validation_plot(data, path)
    except Exception as e:
        print_exception(' ! Unable to create validation plot.', e, show_trance=True)
        
def plot_training(experiment_name):
    try:
        file_controller = FileController(experiment_name)
        path = file_controller.get_training_plot_filepath()    
        data = file_controller.load_training()
        data = np.array(data)
        loss = data[:,0]
        acc = data[:,1]
        plotting_controller = PlottingController(experiment_name)
        plotting_controller.save_training_plot(loss, acc, path)
    except Exception as e:
        print_exception(' ! Unable to create training plot.', e, show_trance=True)

def plot_testing(experiment_name):
    try:
        file_controller = FileController(experiment_name)
        path = file_controller.get_testing_plot_filepath(suffix='all')
        evaluation_controller = EvaluationController(experiment_name)
        acc = evaluation_controller.get_accuracy_list()
        plotting_controller = PlottingController(experiment_name)
        plotting_controller.save_testing_plot(acc, path, title='all')
    except Exception as e:
        print_exception(' ! Unable to create testing plot.', e, show_trance=True)
 
def plot_testing_per_bacteria(experiment_name):
    try:
        file_controller = FileController(experiment_name)
        evaluation_controller = EvaluationController(experiment_name)
        plotting_controller = PlottingController(experiment_name)
        data = evaluation_controller.get_accuracy_list_per_bacteria()
        for bacteria in data.keys():
            acc = data[bacteria]
            path = file_controller.get_testing_plot_filepath(suffix=bacteria)
            plotting_controller.save_testing_plot(acc, path, title=bacteria)
    except Exception as e:
        print_exception(' ! Unable to create testing plot per bacteria.', e, show_trance=True)

def plot_learning_rate(experiment_name):
    try:
        file_controller = FileController(experiment_name)
        path = file_controller.get_learning_rate_plot_filepath()
        data = file_controller.load_training()
        data = np.array(data)
        lr = data[:,4]
        plotting_controller = PlottingController(experiment_name)
        plotting_controller.save_learning_rate_plot(lr, path)
    except Exception as e:
        print_exception(' ! Unable to create learning rate plot.', e, show_trance=True)

def make_report(experiment_name):
    try:
        file_controller = FileController(experiment_name)
        evaluation_controller = EvaluationController(experiment_name)
        report = {
            'mean_accuracy':evaluation_controller.get_accuracy_mean(),
            'mean_accuracy_per_bacteria':evaluation_controller.get_accuracy_mean_per_bacteria(),
            'best_editdistance':evaluation_controller.get_best_validation_loss(),
            'number_of_tested_reads':evaluation_controller.count_reads(),
            'number_of_tested_reads_per_bacteria':evaluation_controller.count_reads_per_bacteria(),
            'unmatched_reads':evaluation_controller.count_unmatched_reads(),
            'unmatched_reads_per_bacteria':evaluation_controller.count_unmatched_reads_per_bacteria(),
            'total_testing_time':evaluation_controller.get_total_testing_time(),
            'total_training_time':evaluation_controller.get_total_training_time(),
            'SMDI':evaluation_controller.get_SMDI(),
            'SMDI_per_bacteria':evaluation_controller.get_SMDI_per_bacteria()
        }
        file_controller.save_evaluation(report)
    except Exception as e:
        print_exception(' ! Unable to create evaluation report.', e, show_trance=True)