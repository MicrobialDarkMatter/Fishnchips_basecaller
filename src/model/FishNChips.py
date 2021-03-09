import os
import numpy as np
import tensorflow as tf

from src.model.ConvBlock import ConvolutionBlock
from src.model.Attention.Transformer import Transformer
from src.model.Attention.Encoder import Encoder

class FishNChips(tf.keras.Model):
    def __init__(self, encoder_blocks, d_model, output_dim, num_heads, dff, max_input_length, max_output_length, dropout):
        super(FishNChips, self).__init__()

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.cnn_block_1 = ConvolutionBlock(filters=d_model/2, kernel=3, dropout_rate=dropout, idx=1)
        self.cnn_block_2 = ConvolutionBlock(filters=d_model, kernel=3, dropout_rate=dropout, idx=2)
        self.encoder = Encoder(encoder_blocks, d_model, num_heads, dff, max_input_length, dropout)
        self.linear = tf.keras.layers.Dense(output_dim)
    
    def call(self, x, training):
        x = self.cnn_block_1(x) 
        x = self.cnn_block_2(x)
        x = self.encoder(x, training)
        x = self.linear(x)
        x = tf.nn.softmax(x)
        return x

    def get_ctc_loss(self, labels, logits):       
        logit_lengths = np.array(logits.shape[0]*[logits.shape[1]])
        label_lengths = np.array(labels.shape[0]*[labels.shape[1]])
        loss = tf.nn.ctc_loss(
            labels, logits, label_lengths, logit_lengths, 
            logits_time_major=False, unique=None)
        loss = tf.reduce_mean(loss)
        return loss