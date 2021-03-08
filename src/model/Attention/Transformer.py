import tensorflow as tf

from src.model.Attention.Encoder import Encoder
from src.model.Attention.Decoder import Decoder

class Transformer(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, output_dim, num_heads, dff, pe_encoder_max_length, pe_decoder_max_length, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_encoder_max_length, rate)
    self.final_layer = tf.keras.layers.Dense(output_dim)

    self.prev_inp = None
    self.cached_enc_output = None
    
  def call(self, x, training, use_cached_enc_ouput=False):         
    x = self._call_encoder(x, training, use_cached_enc_ouput)
    x = self.final_layer(x)  # (batch_size, tar_seq_len, target_vocab_size)
    return x

  def _call_encoder(self, x, training, use_cached_enc_ouput):
    if training: # if we are training we alway run the encoder
      return self.encoder(x, training)  # (batch_size, inp_seq_len, d_model)

    if use_cached_enc_ouput:
      return self.cached_enc_output
        
    out = self.encoder(x, training)  # (batch_size, inp_seq_len, d_model)
    self.cached_enc_output = out
    self.prev_inp = x
    return out