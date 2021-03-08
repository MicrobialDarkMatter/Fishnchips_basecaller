import tensorflow as tf
from src.model.Attention.attention_utils import create_combined_mask
from src.model.FishNChips import FishNChips

class ModelController():
    def __init__(self, config):
        self.model_config = config['model']

    def initialize_model(self):
        print(' - Initializing model.')
        model = FishNChips(
            num_cnn_blocks=self.model_config['cnn_blocks'], 
            max_pool_layer_idx=self.model_config['maxpool_idx'], 
            max_pool_kernel_size=self.model_config['maxpool_kernel'],
            num_layers=self.model_config['attention_blocks'], 
            d_model=self.model_config['d_model'], 
            output_dim=4 + 1, # PAD + ATCG
            num_heads=self.model_config['num_heads'],
            dff=self.model_config['dff'], 
            pe_encoder_max_length=self.model_config['signal_window_size'], 
            pe_decoder_max_length=self.model_config['label_window_size'], 
            rate=self.model_config['dropout_rate'])
        
        x = tf.random.uniform((model.pe_encoder_max_length, 1)) 
        x = tf.expand_dims(x, 0) # add batch size dim -> (batch_size,signal_window_length,1)
        _ = model(x, False)        
        return model