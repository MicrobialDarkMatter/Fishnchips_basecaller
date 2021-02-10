import wandb
import yaml
import src.api as api
import src.data_api as data_api
import tensorflow as tf
from pprint import pprint
from src.utils.data_loader import DataLoader
from src.utils.data_buffer import DataBuffer
from src.utils.data_generator import DataGenerator
from src.model.Attention.CustomSchedule import CustomSchedule
from src.model.Attention.attention_utils import create_combined_mask

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        model_config = build_model_config_from_wandb(config)
        model = api.get_new_model(model_config)
        generator = build_generator(config)

        learning_rate = CustomSchedule(model_config['d_model']*training_config['lr_mult'])
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        for epoch in range(config.epochs):
            train_loss.reset_states()
            batches = next(generator.get_batches(3000))
            for batch,(x,y) in enumerate(batches):
                x = tf.constant(x, dtype=tf.float32)
                y = tf.constant(y, dtype=tf.int32)                
                loss = train_step(x, y, model, loss_object)
                train_loss(loss)
                wandb.log({"loss": train_loss.result(), "epoch": epoch+1})

@tf.function
def train_step(self, x, y, model, loss_object):
    y_input = y[:, :-1]
    y_label = y[:, 1:]
    combined_mask = create_combined_mask(y_input) 
    combined_mask = create_combined_mask(y_input) 
    with tf.GradientTape() as tape:
        y_prediction, _ = model(x, y_input, True, combined_mask)   
        loss = model.get_loss(y_label, y_prediction, loss_object)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def build_model_config_from_wandb(wandb_config):
    return {
        'cnn_blocks':wandb_config.cnn_blocks,
        'maxpool_idx': wandb_config.cnn_blocks // 2,
        'maxpool_kernel': wandb_config.maxpool_kernel,
        'attention_blocks': wandb_config.attention_blocks,
        'd_model': 250,
        'num_heads':wandb_config.num_heads,
        'dff':wandb_config.dff,
        'pe_encoder_max_length':wandb_config.signal_window_size,
        'pe_decoder_max_length':wandb_config.signal_window_size // 3,
        'rate':wandb_config.dropout_rate
    }

def build_generator(wandb_config):
    loader = DataLoader('./data/training.hdf5')
    buffer = DataBuffer(loader, buffer_size=5, batch_size=32, signal_window_size=wandb_config.signal_window_size, signal_window_stride=wandb_config.signal_window_size // 3)
    generator = DataGenerator(buffer, label_window_size = wandb_config.signal_window_size // 3)
    return generator

wandb.login()
with open('./configs/sweeps.yaml', 'r') as f:
    sweep_config = yaml.load(f, Loader=yaml.FullLoader)

sweep_id = wandb.sweep(sweep_config, project="fnch sweep demo")
wandb.agent(sweep_id, train, count=5)