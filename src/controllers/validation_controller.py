import tensorflow as tf
import editdistance
import mappy as mp
import traceback
import time
import math

from src.controllers.inference_controller import InferenceController
from src.utils.base_converter import convert_to_base_string

class ValidationController():
    def __init__(self, config, generator):
        validation_config = config['validation']
        self.batch_size = validation_config['batch_size']
        self.reads = validation_config['reads']
        self.generator = generator
        self.inference_controller = InferenceController()

    def validate(self, model):
        validation_loss = 0
        performed = 0
        start_time = time.time()

        print(' - Starting edit distance validation.')
        for r in range(self.reads):
            print(f' - - validating read {r+1}/{self.reads}', end='\r')
            try:
                x, y_true, read_id = next(self.generator.get_batched_read())
                assert len(x) == len(y_true), f' ! validation of {read_id} has failed - number of signal & label windows varies.'
                
                y_pred = []
                for b in range(0, len(x), self.batch_size):
                    x_batch = x[b:b+self.batch_size]
                    y_batch_pred = self.inference_controller.predict_batch(x_batch, model)
                    y_pred.extend(y_batch_pred)

                total_editdistance = 0
                for pred, true in zip(y_pred, y_true):
                    pred_str = convert_to_base_string(pred, skip_tokens=['S', 'E', 'P'])
                    true_str = convert_to_base_string(true, skip_tokens=['S', 'E', 'P'])
                    total_editdistance += editdistance.eval(pred_str, true_str)
                average_editdistance = total_editdistance / len(y_pred)
                validation_loss += average_editdistance
                performed += 1
            except Exception as e:
                print(e)
                traceback.print_exc()
                       
        print()
        return validation_loss / performed if performed > 0 else -1