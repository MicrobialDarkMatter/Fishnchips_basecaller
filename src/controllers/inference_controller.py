import tensorflow as tf
import numpy as np

# TODO: Use cached decoder output
from src.model.Attention.attention_utils import create_combined_mask

class InferenceController():
    def __init__(self):
        self.start_token = 5
        self.end_token = 6

    """
    Param x: Batch of input signals. Expected shape (batch_size, encoder_max_length, 1), where encoder_max_length is the signal window size
    Param model: Initialized Fish N Chips model
    Returns: List L = Model prediction sequence. Len(L) = batch size. Length of each predition varies
    - This is a non-optimised version, where the loop executes decoder_max_length times
    - For optimised version use 'predict_batch_opt' method, where the loop stops once every example has an output token
    """
    def predict_batch(self, x, model):
        batch_size = x.shape[0]
        y = batch_size*[self.start_token] 
        y = tf.expand_dims(y, 1) # (batch_size, 1)

        for i in range(model.pe_decoder_max_length):
            combined_mask = create_combined_mask(y)
            y_predition, _ = model(x, y, False, combined_mask, False) # (batch_size, i+1, vocab_size)
            y_predition = y_predition[: ,-1:, :] # (batch_size, 1, vocab_size)
            y_predition_base = tf.cast(tf.argmax(y_predition, axis=-1), tf.int32) # (batch_size, 1)
            y = tf.concat([y, y_predition_base], axis=-1) # (batch_size, i+2)

        y = y[:,1:] # (batch_size, encoder_max_length - 1)
        y = self.cut_predition_ends(y) # (batch_size, -)
        return y

    def predict_batch_opt(self, x, model):
        batch_size = x.shape[0]
        y = batch_size*[self.start_token] 
        y = tf.expand_dims(y, 1) # (batch_size, 1)

        end_tokens = np.zeros(batch_size, dtype=int)
        for i in range(model.pe_decoder_max_length):
            combined_mask = create_combined_mask(y)
            y_predition, _ = model(x, y, False, combined_mask, False) # (batch_size, i+1, vocab_size)
            y_predition = y_predition[: ,-1:, :] # (batch_size, 1, vocab_size)
            y_predition_base = tf.cast(tf.argmax(y_predition, axis=-1), tf.int32) # (batch_size, 1)
            
            for j in range(batch_size):
                if y_predition_base[j][0] == self.end_token:
                    end_tokens[j] = 1
            
            if all(j == 1 for j in end_tokens):
                break
            
            y = tf.concat([y, y_predition_base], axis=-1) # (batch_size, i+2)
        y = y[:,1:] # (batch_size, encoder_max_length - 1)
        y = self.cut_predition_ends(y) # (batch_size, -)
        return y

    """
    Param y - Model predictions for each example in a batch
    Returns - List of predictions, each cut after the end token
    """
    def cut_predition_ends(self, y):
        y_out = []
        for y_example in y:
            for i,token in enumerate(y_example):
                if token == self.end_token or i == len(y_example)-1:
                    y_out.append(y_example[:i].numpy())
                    break
        return y_out