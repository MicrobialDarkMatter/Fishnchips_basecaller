import tensorflow as tf
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
        
        learning_rate = CustomSchedule(model_config['d_model']*training_config['lr_mult'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none') 

    def train(self):
        print(' - Training model.')
        waited = 0
        validation_loss= 1e10 if self.results == [] else self.get_best_validation_loss()
        print(f' - Initial validation loss: {validation_loss}.')
        model_weights = self.model.get_weights()

        for epoch in range(self.epochs):
            waited = 0 if epoch < self.warmup else waited
            start_time = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            batches = next(self.generator.get_batches(self.batches))
            for batch,(x,y) in enumerate(batches):
                x = tf.constant(x, dtype=tf.float32)
                y = tf.constant(y, dtype=tf.int32)
                self.train_step(x, y)
                print (f' - - Epoch:{epoch+1}/{self.epochs} | Batch:{batch+1}/{len(batches)} | Loss:{self.train_loss.result():.4f} | Accuracy:{self.train_accuracy.result():.4f}', end="\r")
            print()

            current_validation_loss = self.validation_controller.validate(self.model)
            self.results.append([self.train_loss.result(), self.train_accuracy.result(), validation_loss, time.time()])
            self.file_controller.save_training(self.results)            
            print (f' = = Epoch:{epoch+1}/{self.epochs} | Loss:{self.train_loss.result():.4f} | Accuracy:{self.train_accuracy.result():.4f} | Validation loss:{validation_loss} | Took:{time.time() - start_time} secs')

            if current_validation_loss < validation_loss:
                waited = 0
                validation_loss = current_validation_loss
                print(' - - Model validation accuracy improvement - saving model weights.')
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
        y_input = y[:, :-1]
        y_label = y[:, 1:]
    
        combined_mask = create_combined_mask(y_input) 
        with tf.GradientTape() as tape:
            y_prediction, _ = self.model(x, y_input, True, combined_mask)   
            loss = self.model.get_loss(y_label, y_prediction, self.loss_object)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(y_label, y_prediction)

    def get_best_validation_loss(self):
        validation_losses = self.results[:,2]
        min_validation_loss = 1e10
        for validation_loss in validation_losses:
            if validation_loss < 0:
                continue
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
        return min_validation_loss