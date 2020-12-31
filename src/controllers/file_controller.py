import os

class File_Controller():
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