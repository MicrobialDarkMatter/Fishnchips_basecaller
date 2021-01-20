import os 
import sys
import json
import inquirer

class UIController():
    def __init__(self, config, experiment_name):
        self.config = config
        self.path = f'./trained_models/{experiment_name}'
        self.retrain = True
        self.retest = True
        self.append_test = False
        
    def ask_retrain(self):
        if os.path.exists(f'{self.path}/model.h5') == False:
            print(' - Model will be trained as a trained model was not found.')
            return
        message = 'A trained model already exists, would you like to retrain it?'
        choices = ['retrain', 'skip training']
        question = inquirer.List('retrain', message, choices)
        answer = inquirer.prompt([question])
        self.retrain = answer['retrain'] == 'retrain'

    def ask_retest(self):
        if os.path.exists(f'{self.path}/evaluation.json') == False:
            print(' - Model will be tested as its evaluation was not found.')
            return
        message = 'A trained model already exists, would you like to retrain it?'
        choices = ['re-evaluate', 'append to existing', 'skip evaluation']
        question = inquirer.List('retest', message, choices)
        answer = inquirer.prompt([question])
        self.retest = answer['retest'] in ['re-evaluate', 'append to existing']
        self.append_test = answer['retest'] == 'append to existing'

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