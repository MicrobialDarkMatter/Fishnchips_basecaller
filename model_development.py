import numpy as np
import src.api as api
import tensorflow as tf
import src.data_api as data_api
from src.utils.config_loader import load_config
from src.model.Attention.attention_utils import create_combined_mask

def setup():
    config_path = './configs/local.json'
    return load_config(config_path), 'debug'

def get_data():
    generator = data_api.get_generator(config)
    x,y  = next(generator.get_batch())
    return x, y

config, experiment_name = setup()
model = api.get_new_model(config)
x,y = get_data()


logits = model(x,y,False,None,False) #(1,50,7) == (Batch size, frmaes, no labels)
labels = y
label_length = np.array(1*[100])
logit_length = np.array(1*[50])

print(logits.shape)
print(labels.shape)
print(logit_length)
print(label_length)
out = tf.nn.ctc_loss(
    labels, logits, label_length, logit_length, 
    logits_time_major=False, unique=None,blank_index=-1)




# y = tf.nn.softmax(y)
# print(y)
# y = tf.cast(tf.argmax(y, axis=-1), tf.int32)
# print(y)
# print(y.shape)
