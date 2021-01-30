import os 
import sys
import json
import inquirer
from src.controllers.file_controller import FileController

class UIController():
    def __init__(self, config, experiment_name):
        self.config = config
        self.file_controller = FileController(experiment_name)

        self.skip_training = False
        self.discard_training = False
        self.continue_training = True 

        self.skip_testing = False
        self.discard_testing = False 
        self.continue_testing = True
        
    def ask_retrain(self):
        if self.file_controller.trained_model_exists() == False:
            print(' - Trained model not found. Model will be trained.')
            return
        message = 'A trained model already exists, would you like to retrain it?'
        choices = ['skip training', 'continue training existing model', 'discard existing model']
        question = inquirer.List('retrain', message, choices)
        answer = inquirer.prompt([question])
        
        self.skip_training = answer['retrain'] == 'skip training'
        self.discard_training = answer['retrain'] == 'discard existing model'
        self.continue_training = answer['retrain'] == 'continue training existing model'
        
    def ask_retest(self):
        if self.file_controller.evaluation_exists() == False:
            print(' - Testing results not found. Model will be tested.')
            return
        message = 'Model evaluation already exists, what would you like to do?'
        choices = ['skip testing', 'append to existing results', 'discard existing results']
        question = inquirer.List('retest', message, choices)
        answer = inquirer.prompt([question])

        self.skip_testing = answer['retest'] == 'skip testing'
        self.discard_testing = answer['retest'] == 'discard existing results'
        self.continue_testing = answer['retest'] == 'append to existing results'

    def ask_parameters(self):
        message = 'Continue with these experiment parameters?'
        choices = ['yes', 'no']
        question = inquirer.List('parameters', message, choices)
        answer = inquirer.prompt([question])
        if answer['parameters'] == 'no':
            sys.exit()

    def print_parameters(self, key):
        print(f' - Verify {key} parameters')
        print(json.dumps(self.config[key], indent=4, sort_keys=True))    