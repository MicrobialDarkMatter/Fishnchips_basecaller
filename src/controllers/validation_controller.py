import tensorflow as tf
import editdistance
import mappy as mp
import traceback
import time
import math

from src.utils.base_converter import convert_to_base_string
from src.model.Attention.attention_utils import create_combined_mask

class ValidationController():
    def __init__(self, config, generator):
        validation_config = config['validation']
        self.batches = 300
        self.generator = generator

    def validate(self, model):
        start_time = time.time()
        batches = next(self.generator.get_batches(self.batches))
        loss_list = []
        for batch,(x,y) in enumerate(batches):
            x = tf.constant(x, dtype=tf.float32)
            y = tf.constant(y, dtype=tf.int32) 
            loss = self.validation_step(x, y, model)
            loss_list.append(loss)
        return np.array(loss_list).mean()

    @tf.function
    def validation_step(self, x, y, model):
        y_input = y[:, :-1]
        y_label = y[:, 1:]
        combined_mask = create_combined_mask(y_input) 
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none') 
        with tf.GradientTape() as tape:
            y_prediction, _ = model(x, y_input, True, combined_mask)   
            loss = model.get_loss(y_label, y_prediction, loss_object)
        return loss
