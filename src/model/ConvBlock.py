import tensorflow as tf

class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, dropout_rate, idx):
        super(ConvolutionBlock, self).__init__()
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.cnn_layer = tf.keras.layers.Conv1D(filters, kernel, padding="same", activation=None, strides=2, name=f'conv_{idx}')
        self.bn_layer = tf.keras.layers.BatchNormalization(name=f'bn_{idx}')
        self.activation_layer = tf.keras.layers.Activation('relu', name=f'activation_{idx}')

    def call(self, x):
        x = self.dropout_layer(x)
        x = self.cnn_layer(x)
        x = self.bn_layer(x)
        x = self.activation_layer(x)
        return x