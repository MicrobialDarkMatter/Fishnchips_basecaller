import tensorflow as tf
import editdistance
import mappy as mp
import time
import math

from src.controllers.data_controller import DataController
from src.controllers.inference_controller import InferenceController
from src.utils.validation_algorithm import VALIDATION_ALGORITHM

class ValidationController():
    def __init__(self, validation_config):
        self.batch_size = validation_config['batch_size']
        self.algorithm = validation_config['algorithm']
        self.data_controller = DataController(validation_config['data'], self.batch_size)
        self.inference_controller = InferenceController()

    def validate(self, model):
        print(' - Validating model.')
        if self.algorithm == VALIDATION_ALGORITHM.editdistance:
            return self.validate_editdistance(model)
        if self.algorithm == VALIDATION_ALGORITHM.mappy:
            return self.validate_mappy(model)
        raise f' ! Unknown validation algorithm. Valid options are: {VALIDATION_ALGORITHM.get_options()}'

    def validate_editdistance(self, model):
        x,y = self.data_controller.load_data(cap=self.batch_size*3)
        batches = math.ceil(x.shape[0] / self.batch_size)
        validation_dataset = self.data_controller.process_data(x, y)

        print(' - - Starting...')
        validation_loss = 0
        performed = 0
        start_time = time.time()
        for batch,(x,y) in enumerate(validation_dataset):
            x = tf.constant(x, dtype=tf.float32)
            y = tf.constant(y, dtype=tf.int32)
            y_label = y[:, 1:]
            y_prediction = self.inference_controller.predict_batch_opt(x, model)
            
            total_editdistance = 0
            for prediction, label in zip(y_prediction, y_label):
                total_editdistance += editdistance.eval(prediction, label.numpy())
            average_editdistance = total_editdistance / self.batch_size
            validation_loss += average_editdistance
            print (f' - - Batch:{batch+1}/{batches} | Validation loss:{validation_loss} | Current average edit distance:{average_editdistance}.', end="\r")
        print()
        return validation_loss

    def validate_mappy(self, model):
        raise 'Not implemented.'