import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, \
    Flatten, Reshape, MaxPool2D, UpSampling2D,Conv2DTranspose,\
    Add, Concatenate, Activation, Softmax, LayerNormalization
import numpy as np
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import activations
# from tensorflow.keras.applications import ResNet50, MobileNetV2, InceptionV3
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras import activations

import config_model as cfg
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import add, Activation
# from tensorflow.keras.regularizers import l2

# max_group = cfg.max_group
# max_pillars = cfg.max_pillars
nb_channels = cfg.nb_channels
batch_size = cfg.BATCH_SIZE
image_size = cfg.img_shape
nb_anchors = cfg.nb_anchors
nb_classes = cfg.nb_classes


def SimpleConv2D(entry_layer, out_features, Ksize, stride=1, group=1, pad='same'):
    c = Conv2D(out_features,
               kernel_size=(Ksize, Ksize),
               strides=(stride, stride),
               groups=group,
               kernel_initializer = tf.keras.initializers.HeNormal(), 
               padding=pad)(entry_layer)
    return c


def BottleNeckBlock(entry_layer, in_features, out_features, exp, stride=1):
    expanded_features = out_features * exp

    conv_seq = SimpleConv2D(entry_layer, in_features,
                            Ksize=5, group=in_features,
                            stride=stride)

    conv_seq = LayerNormalization()(conv_seq)

    conv_seq = SimpleConv2D(conv_seq, expanded_features, Ksize=1, stride=1)
    conv_seq = Activation('relu')(conv_seq)

    conv_seq = SimpleConv2D(conv_seq, out_features, Ksize=1, stride=1)

    #     conv_shortcut = SimpleConv2D(entry_layer,out_features, Ksize = 1,stride = stride)

    out = Add()([conv_seq, entry_layer])
    #     out = Activation('relu')(add)

    return out
def conv_block(entrylayer, channels_out, kernelsize, stride, pad):
    c = Conv2D(channels_out,kernel_size=kernelsize, strides=stride,
               kernel_initializer = tf.keras.initializers.HeNormal(), 
               kernel_regularizer=l2(0.01),
               padding = pad, use_bias = False)(entrylayer)
    c = BatchNormalization()(c)
    c = Activation('relu')(c)
    return c
def conv_block_up(entrylayer, channels, kernelsize, stride, pad = 'same'):
    c = Conv2DTranspose(filters = channels,
                        kernel_size = kernelsize,
                        strides = stride,
                        padding=pad,
                        kernel_regularizer=l2(0.01),
                        kernel_initializer = tf.keras.initializers.HeNormal())(entrylayer)
    c = BatchNormalization()(c)
    c = Activation('relu')(c)
    return c

def Head(entry_layer):

    last_class = Conv2D(nb_anchors, (1, 1))(entry_layer)
    last_class = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    last_conf = Conv2D(256, (1, 1), padding = 'valid', name="occ1", activation="relu")(entry_layer)
    last_conf = Conv2D(nb_anchors, (1, 1), name="occ2", activation="sigmoid")(last_conf)
    last_conf = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, 1), name="F_occ")(last_conf)

    last_pos = Conv2D(64, (1, 1), padding = 'valid', name="loc1", activation="relu")(entry_layer)
    last_pos = Conv2D(nb_anchors*3, (1, 1), name="loc2")(last_pos)
    last_pos =  Reshape(tuple(i//cfg.factor for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    last_dim = Conv2D(64, (1, 1), padding = 'valid', name="size1", activation="relu")(entry_layer)
    last_dim = Conv2D(nb_anchors*3, (1, 1), name="size2")(last_dim)
    last_dim = Reshape(tuple(i//cfg.factor for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    last_rot = Conv2D(32, (1, 1), padding = 'valid', name="rot1", activation="relu")(entry_layer)
    last_rot = Conv2D(nb_anchors, (1, 1), name="rot2")(last_rot)
    last_rot = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, 1), name="F_rot")(last_rot)

    output = Concatenate()([last_conf,last_pos,last_dim,last_rot,last_class])


    return output

def Img2BEV(img):
    c1 = SimpleConv2D(img, out_features=32,
                     Ksize=3, stride=1, group=1, pad='same')
    # c1 = BottleNeckBlock(c1, 64, 64, 4, stride=1)
    for i in range(1):
        c1 = BottleNeckBlock(c1, 32, 32, 2, stride=1)


    c2 = LayerNormalization()(c1)
    c2 = SimpleConv2D(c2, out_features=64,
                     Ksize=3, stride=2, group=1, pad='same')
    # c2 = BottleNeckBlock(c2, 128, 128, 4, stride=1)

    for i in range(1):
        c2 = BottleNeckBlock(c2, 64, 64, 2, stride=1)


    c3 = LayerNormalization()(c2)
    c3 = SimpleConv2D(c3, out_features=64,
                     Ksize=3, stride=2, group=1, pad='same')
    # c3 = BottleNeckBlock(c3, 128, 128, 4, stride=1)

    for i in range(2):
        c3 = BottleNeckBlock(c3, 64, 64, 2, stride=1)


    c4 = LayerNormalization()(c3)
    c4 = SimpleConv2D(c4, out_features=128,
                     Ksize=3, stride=2, group=1, pad='same')
    # c4 = BottleNeckBlock(c4, 256, 256, 4, stride=1)

    for i in range(3):
        c4 = BottleNeckBlock(c4, 128, 128, 2, stride=1)


    c5 = LayerNormalization()(c4)
    c5 = SimpleConv2D(c5, out_features=128,
                     Ksize=3, stride=1, group=1, pad='same')
    # c5 = BottleNeckBlock(c5, 256, 256, 4, stride=1)

    for i in range(3):
        c5 = BottleNeckBlock(c5, 128, 128, 2, stride=1)

    # UP step
        
    # C2 - 256 // C3 - 128 // C4 - 64 // C5 - 64
    
    c_5_4 = Concatenate()([c4,c5]) # 64.64.512c
    c_54 = conv_block_up(c_5_4, 128, (3,3), (2,2)) # 128.128.256
    c3_54 = Concatenate()([c3,c_54])# 128.128.384c
    c_543 = conv_block_up(c3_54, 128, (3,3), (2,2)) # 256.256.256
    c2_543 = Concatenate()([c2,c_543])# 256.256.384c

    out = LayerNormalization()(c2_543)
    # out = SimpleConv2D(out, out_features=256,
    #                  Ksize=3, stride=1, group=1, pad='same')
    # out = BottleNeckBlock(out, 256, 256, 2, stride=1)
    # print(out.shape)
    # exit()
    return out

def My_Model(input_img):
    # Enter the pillar and pillar mean
    # Convolutional Network feature
    Fimg = Img2BEV(input_img)
    # Head detection
    out = Head(Fimg)
    return out


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model
    # from config_model import cfg
    # input_pillar_shape = (12000, 100, 9)
    # input_pillar_mean_shape = (12000, 3)
    input_img_shape = (512,512,3)


    # input_pillar = Input(input_pillar_shape, batch_size=batch_size)
    # input_pillar_mean = Input(input_pillar_mean_shape, batch_size=batch_size)
    input_img = Input(input_img_shape,batch_size = batch_size)

    output = My_Model(input_img)
    model = Model(inputs=input_img, outputs=output)
    model.summary()
    # tf.keras.utils.plot_model(
    # model,
    # to_file="D:/SCRIPTS/Doc_lidar_v2/models/model2.png",
    # show_shapes=True,
    # dpi = 300)