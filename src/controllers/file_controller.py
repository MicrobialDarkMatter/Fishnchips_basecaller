import os
import json
import numpy as np

class FileController():
    def __init__(self, experiment_name):
        self.path = f'./trained_models/{experiment_name}'
    
    def get_model_filepath(self):
        return f'{self.path}/model.h5'
    
    def get_training_filepath(self):
        return f'{self.path}/training.npy'

    def get_testing_filepath(self):
        return f'{self.path}/testing.json'

    def get_assembly_directory_path(self):
        return f'{self.path}/assemblies'

    def get_assembly_filepath(self, read_id, iteration, bacteria):
        return f'{self.get_assembly_directory_path()}/{iteration}_{bacteria}_{read_id}.txt'

    def get_validation_plot_filepath(self):
        return f'{self.path}/report/validation.png'

    def get_training_plot_filepath(self):
        return f'{self.path}/report/training.png'

    def get_testing_plot_filepath(self, suffix):
        return f'{self.path}/report/testing_{suffix}.png'

    def get_evaluation_filrpath(self):
        return f'{self.path}/report/evaluation.json'

    def trained_model_exists(self):
        return os.path.exists(self.get_model_filepath())

    def testing_result_exists(self):
        return os.path.exists(self.get_testing_filepath())

    def training_result_exists(self):
        return os.path.exists(self.get_training_filepath())

    def create_experiment_dir(self):
        try: 
            if os.path.exists(self.path):
                return
            os.makedirs(self.path)
        except Exception as e:
            print(e)
            print(f' ! Error occured when creating experiment run directory. Path: {path}')

    def create_assembly_directory(self):
        try:
            assembly_path = f'{self.path}/assemblies' 
            if os.path.exists(assembly_path):
                return 
            os.makedirs(assembly_path)
        except Exception as e:
            print(e)
            print(f' ! Error occured when creating experiment assembly directory. Path: {assembly_path}')    

    def create_report_directory(self):
        try:
            report_path = f'{self.path}/report'
            if os.path.exists(report_path):
                return
            os.makedirs(report_path)
        except Exception as e:
            print(e)
            print(f' ! Error occured when creating experiment report directory. Path: {report_path}')    

    def load_testing(self):
        assert self.testing_result_exists(), f'Evaluation was requested, but {self.get_testing_filepath()} does not exist.'
        with open(self.get_testing_filepath(), 'r') as f:
            return json.load(f)

    def load_training(self):
        assert self.training_result_exists(), f'Training result was requested, but {self.get_training_filepath()} does not exist.'
        training = np.load(self.get_training_filepath())    
        training = training.tolist()
        return training    

    def save_testing(self, evaluation):
        with open(self.get_testing_filepath(), 'w') as f:
            json.dump(evaluation, f, indent=4)

    def save_training(self, results):
        path = self.get_training_filepath()
        np.save(path, results)

    def save_evaluation(self, report):
        path = self.get_evaluation_filrpath()
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)

    def save_model(self, model):
        path = self.get_model_filepath()
        model.save_weights(path)

    def teardown_evaluation(self):
        path = self.get_testing_filepath()
        if os.path.exists(path):
            print(' ! Removing existing evaluation.')
            os.remove(path)

    def teardown_training(self):
        path = self.get_training_filepath()
        if os.path.exists(path):
            print(' ! Removing existing training progress data.')
            os.remove(path)

    def teardown_model(self):
        path = self.get_model_filepath()
        if os.path.exists(path):
            print(' ! Removing trained model.')
            os.remove(path)
        
    def teardown_assemblies(self):
        directory = self.get_assembly_directory_path()
        assembly_files = os.listdir(directory)
        if assembly_files != []:
            print(' ! Removing assemblies.')
            for filename in assembly_files:
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
