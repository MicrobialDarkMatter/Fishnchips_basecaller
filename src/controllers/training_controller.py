import tensorflow as tf
import numpy as np
import time
import math
import sys
import os

from src.controllers.data_controller import DataController
from src.controllers.validation_controller import ValidationController
from src.model.Attention.CustomSchedule import CustomSchedule
from src.model.Attention.attention_utils import create_combined_mask

class Training_Controller():
    def __init__(self, config, experiment_name, model, retrain=True):
        training_config = config['training']
        model_config = config['model']
        
        self.epochs = training_config['epochs']
        self.patience = training_config['patience']
        self.warmup = training_config['warmup']
        self.batch_size = training_config['batch_size']
        self.model = model
        self.retrain = retrain
        
        self.model_path = f'./trained_models/{experiment_name}/model.h5'
        self.training_path = f'./trained_models/{experiment_name}/training.npy'

        learning_rate = CustomSchedule(model_config['d_model']*training_config['lr_mult'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none') 

        self.data_controller = DataController(training_config['data'], self.batch_size)
        self.validation_controller = ValidationController(config['validation'])

    def load_model(self):
        print(' - Skipping training and loading a trained model from a file.')
        self.model.load_weights(self.model_path)
        return self.model

    def train(self):
        if self.retrain == False:
            return self.load_model()

        print(' - Training model.')
        accuracies = []
        old_validation_loss= 1e10
        model_weights = None

        x, y = self.data_controller.load_data()
        batches = math.ceil(x.shape[0] / self.batch_size)
        training_dataset = self.data_controller.process_data(x, y)
        print(' - - Starting...')
        for epoch in range(self.epochs):
            start_time = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for batch,(x,y) in enumerate(training_dataset):
                x = tf.constant(x, dtype=tf.float32)
                y = tf.constant(y, dtype=tf.int32)
                self.train_step(x, y)
                print (f' - - Epoch:{epoch+1}/{self.epochs} | Batch:{batch+1}/{batches} | Loss:{self.train_loss.result():.4f} | Accuracy:{self.train_accuracy.result():.4f}', end="\r")
            print()

            validation_loss = self.validation_controller.validate(self.model)
            accuracies.append([self.train_loss.result(), self.train_accuracy.result(), validation_loss, time.time()])
            np.save(self.training_path, np.array(accuracies)) 
            print (f' = = Epoch:{epoch+1}/{self.epochs} | Loss:{self.train_loss.result():.4f} | Accuracy:{self.train_accuracy.result():.4f} | Validation loss:{validation_loss} | Took:{time.time() - start_time} secs')

            if validation_loss < old_validation_loss:
                # TODO: Add waiting
                old_validation_loss = validation_loss
                print(' - - Model validation accuracy improvement - saving model weights.')
                self.model.save_weights(self.model_path)
                model_weights = self.model.get_weights()
 
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