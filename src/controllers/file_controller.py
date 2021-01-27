import os
import json

class FileController():
    def __init__(self, experiment_name):
        self.path = f'./trained_models/{experiment_name}'
    
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

    def load_evaluation(self):
        assert self.evaluation_exists(), f'Evaluation was requested, but {self.get_evaluation_filepath()} does not exist.'
        with open(self.get_evaluation_filepath(), 'r') as f:
            return json.load(f)

    def save_evaluation(self, evaluation):
        with open(self.get_evaluation_filepath, 'w') as f:
            json.dump(evaluation, f, indent=4)

    def get_model_filepath(self):
        return f'{self.path}/model.h5'
    
    def get_training_filepath(self):
        return f'{self.path}/training.npy'

    def get_evaluation_filepath(self):
        return f'{self.path}/evaluation.json'

    def get_assembly_filepath(self, read_id, iteration):
        return f'{self.path}/assemblies/read_{iteration}_id_{read_id}' 

    def trained_model_exists(self):
        return os.path.exists(self.get_model_filepath())

    def evaluation_exists(self):
        return os.path.exists(self.get_evaluation_filepath())