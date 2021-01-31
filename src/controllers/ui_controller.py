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
        self.new_training = False
        self.continue_training = True 

        self.skip_testing = False
        self.new_testing = False 
        self.continue_testing = True
    
    def ask_retrain(self):
        if self.file_controller.trained_model_exists() == False:
            print(' - Trained model not found. Model will be trained.')
            self.continue_training = False
            self.skip_training = False
            self.new_training = True
            return
        message = 'A trained model already exists, would you like to retrain it?'
        choices = ['skip training', 'continue training existing model', 'new training (discard existing)']
        question = inquirer.List('retrain', message, choices)
        answer = inquirer.prompt([question])
        
        self.skip_training = answer['retrain'] == 'skip training'
        self.new_training = answer['retrain'] == 'new training (discard existing)'
        self.continue_training = answer['retrain'] == 'continue training existing model'
        
    def ask_retest(self):
        if self.file_controller.evaluation_exists() == False:
            print(' - Testing results not found. Model will be tested.')
            self.continue_testing = False 
            self.skip_testing = False 
            self.new_testing = True
            return
        message = 'Model testing results already exist, what would you like to do?'      
        choices = self.get_retest_choices()
        question = inquirer.List('retest', message, choices)
        answer = inquirer.prompt([question])

        self.skip_testing = answer['retest'] == 'skip testing'
        self.new_testing = answer['retest'] == 'new testing (discard existing)'
        self.continue_testing = answer['retest'] == 'append to existing results'

    def get_retest_choices(self):
        if self.new_training or self.continue_training:
            return ['skip testing', 'new testing (discard existing)'] # Do not allow to append testing analysis if model is retrained or improved
        else:
            return ['skip testing', 'append to existing results', 'new testing (discard existing)']

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