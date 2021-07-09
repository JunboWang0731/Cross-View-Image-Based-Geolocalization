import tensorflow as tf
from VGG import *
from ResNet import *

# 'VGG16'
def siamese_VGG16(x_sat, x_grd):
    print("VGG16!")
    sat_model = VGG16()
    sat_vgg = sat_model.VGG16Conv(x_sat, 'VGG16_sat')
    sat_vgg = tf.contrib.layers.flatten(sat_vgg)

    grd_model = VGG16()
    grd_vgg = grd_model.VGG16Conv(x_grd, 'VGG16_grd')
    grd_vgg = tf.contrib.layers.flatten(grd_vgg)

    return tf.nn.l2_normalize(sat_vgg, dim=1), tf.nn.l2_normalize(grd_vgg, dim=1)

# 'ResNet50'
def siamese_ResNet50(x_sat, x_grd):
    print("ResNet50!")
    sat_model = ResNet50()
    sat_resnet = sat_model.ResNet50Conv(x_sat, 'ResNet50_sat')
    sat_resnet = Flatten()(sat_resnet)

    grd_model = ResNet50()
    grd_resnet = grd_model.ResNet50Conv(x_grd, 'ResNet50_grd')
    grd_resnet = Flatten()(grd_resnet)

    return tf.nn.l2_normalize(sat_resnet, dim=1), tf.nn.l2_normalize(grd_resnet, dim=1)