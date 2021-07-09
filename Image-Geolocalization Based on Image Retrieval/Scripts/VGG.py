import tensorflow as tf

class VGG16:
    # Define kernels
    def Conv2D(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def MaxPool2D(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Define layers
    def Conv2DLayer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                  name='layer', activation_function=tf.nn.relu):
        with tf.variable_scope(name):

            weight = tf.get_variable(name='weight', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim], trainable=trainable,
                                   initializer=tf.contrib.layers.xavier_initializer())

            if activated:
                out = activation_function(self.Conv2D(x, weight) + bias)
            else:
                out = self.Conv2D(x, weight) + bias

            return out

    def MaxPool2DLayer(self, x, name):
        with tf.variable_scope(name):
            out = self.MaxPool2D(x)
            return out

    def Dense(self, x, input_dim, output_dim, init_dev, init_bias, name):
        with tf.name_scope(name):
            weight = tf.get_variable(name='weights', shape=[input_dim, output_dim],
                                     trainable=True,
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=init_dev))
            bias = tf.get_variable(name='bias', shape=[output_dim],
                                   trainable=True,
                                   initializer=tf.constant_initializer(init_bias))
            out = tf.nn.xw_plus_b(x, weight, bias)
        return out

    # VGG16
    def VGG16Conv(self, input, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            x = self.Conv2DLayer(input, 3, 3, 64, True, True, 'conv1_1')
            x = self.Conv2DLayer(x, 3, 64, 64, True, True, 'conv1_2')
            x = self.MaxPool2DLayer(x, 'conv1_maxpool')

            x = self.Conv2DLayer(x, 3, 64, 128, True, True, 'conv2_1')
            x = self.Conv2DLayer(x, 3, 128, 128, True, True, 'conv2_2')
            x = self.MaxPool2DLayer(x, 'conv2_maxpool')

            x = self.Conv2DLayer(x, 3, 128, 256, True, True, 'conv3_1')
            x = self.Conv2DLayer(x, 3, 256, 256, True, True, 'conv3_2')
            x = self.Conv2DLayer(x, 3, 256, 256, True, True, 'conv3_3')
            x = self.MaxPool2DLayer(x, 'conv3_maxpool')

            x = self.Conv2DLayer(x, 3, 256, 512, True, True, 'conv4_1')
            x = self.Conv2DLayer(x, 3, 512, 512, True, True, 'conv4_2')
            x = self.Conv2DLayer(x, 3, 512, 512, True, True, 'conv4_3')
            x = self.MaxPool2DLayer(x, 'conv4_maxpool')

            x = self.Conv2DLayer(x, 3, 512, 512, True, True, 'conv5_1')
            x = self.Conv2DLayer(x, 3, 512, 512, True, True, 'conv5_2')
            x = self.Conv2DLayer(x, 3, 512, 512, True, True, 'conv5_3')
            output = self.MaxPool2DLayer(x, 'conv5_maxpool')

            return output





