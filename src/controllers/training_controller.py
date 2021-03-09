import tensorflow as tf
import numpy as np
import time
import math
import sys
import os

from src.controllers.file_controller import FileController
from src.controllers.validation_controller import ValidationController
from src.model.Attention.CustomSchedule import CustomSchedule
from src.model.Attention.attention_utils import create_combined_mask
from src.utils.data_generator import DataGenerator
from src.utils.data_buffer import DataBuffer
from src.utils.data_loader import DataLoader
from src.utils.base_converter import convert_to_ctc_base_string, convert_to_base_string

class TrainingController():
    def __init__(self, config, experiment_name, model, generator, validation_controller, new_training):
        training_config = config['training']
        model_config = config['model']
        
        self.generator = generator
        self.validation_controller = validation_controller
        
        self.file_controller = FileController(experiment_name)
        self.results = [] if new_training else self.file_controller.load_training()

        self.epochs = training_config['epochs']
        self.patience = training_config['patience']
        self.warmup = training_config['warmup']
        self.batches = training_config['batches']
        self.batch_size = training_config['batch_size']
        self.model = model
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-2)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def train(self):
        print(' - Training model.')
        waited = 0
        
        min_loss = 1e10 if self.results == [] else self.set_loss()
        print(f' - Initial loss: {min_loss}.')
        model_weights = self.model.get_weights()

        for epoch in range(self.epochs):
            waited = 0 if epoch < self.warmup else waited
            start_time = time.time()
            self.train_loss.reset_states()

            batches = next(self.generator.get_batches(self.batches))
            for batch,(x,y) in enumerate(batches):
                x = tf.constant(x, dtype=tf.float32)
                y = tf.convert_to_tensor(y, dtype=tf.int32)                
                self.train_step(x, y)
                print (f' - - Epoch:{epoch+1}/{self.epochs} | Batch:{batch+1}/{len(batches)} | Loss:{self.train_loss.result():.4f} ', end="\r")
            print()

            # self.validation_controller.validate(self.model)
            
            lr = self.get_current_learning_rate()
            self.results.append([self.train_loss.result(), -1, time.time(), lr])
            self.file_controller.save_training(self.results)            
            self.print_sample_output()
            print (f' = = Epoch:{epoch+1}/{self.epochs} | Loss:{self.train_loss.result():.4f} | Took:{time.time() - start_time} secs | Learning rate:{lr:.10}')

            if self.train_loss.result() < min_loss:
                waited = 0
                min_loss = self.train_loss.result()
                print(' - - Model accuracy improvement - saving model weights.')
                self.file_controller.save_model(self.model)                
                model_weights = self.model.get_weights()
            else:
                waited += 1
                if waited > self.patience:
                    print(f' - Stopping training ( out of patience - model has not improved for {self.patience} epochs.')
                    break
            
        self.model.set_weights(model_weights)
        return self.model
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_prediction = self.model(x, training=True)   
            loss = self.model.get_ctc_loss(y, y_prediction)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)

    def print_sample_output(self):
        batches = next(self.generator.get_batches(1))
        for batch,(x,y) in enumerate(batches):
            x = tf.constant(x[:1], dtype=tf.float32)
            y = tf.constant(y[:1], dtype=tf.int32)
            p = self.model(x, training=True)
            p = tf.transpose(p, [1, 0, 2])
            p, _ = tf.nn.ctc_greedy_decoder(p, np.array(1*[self.model.max_input_length/4]), merge_repeated=False)
            p = p[0].values.numpy()
            print(' = = Sample output:')
            print(f' = = predicted: {convert_to_ctc_base_string(p)} | {p.shape}')
            print(f' = = true     : {convert_to_base_string(y.numpy()[0])}')
            print(60*'-')


    def set_loss(self):
        results = np.array(self.results)
        losses = results[:,0]
        min_loss = 1e10
        for loss in losses:
            if loss < min_loss:
                min_loss = loss
        return min_loss

    def get_current_learning_rate(self):
        lr = self.optimizer._decayed_lr("float32").numpy()
        return float(lr)

