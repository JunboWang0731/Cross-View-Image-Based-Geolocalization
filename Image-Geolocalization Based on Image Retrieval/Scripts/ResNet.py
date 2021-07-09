from tensorflow.keras.layers import Input, \
    Conv2D, MaxPool2D, BatchNormalization, ReLU, \
    Add, GlobalAvgPool2D, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
import tensorflow as tf

class ResNet50:
    def conv_batchnorm_relu(self, x, filters, kernel_size, strides=1):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def identity_block(self, tensor, filters):
        x = self.conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
        x = self.conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
        x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
        x = BatchNormalization()(x)

        x = Add()([tensor, x])
        x = ReLU()(x)

        return x

    def projection_block(self, tensor, filters, strides):
        # left stream
        x = self.conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
        x = self.conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
        x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
        x = BatchNormalization()(x)

        # right stream
        shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([shortcut, x])
        x = ReLU()(x)

        return x

    def resnet_block(self, x, filters, reps, strides):
        x = self.projection_block(x, filters, strides)
        for _ in range(reps-1):
            x = self.identity_block(x, filters)
        return x

    def ResNet50Conv(self, input, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            x = self.conv_batchnorm_relu(input, filters=64, kernel_size=7, strides=2)
            x = MaxPool2D(pool_size = 3, strides=2)(x)
            x = self.resnet_block(x, filters=64, reps=3, strides=1)
            x = self.resnet_block(x, filters=128, reps=4, strides=2)
            x = self.resnet_block(x, filters=256, reps=6, strides=2)
            output = self.resnet_block(x, filters=512, reps=3, strides=2)

            return output


