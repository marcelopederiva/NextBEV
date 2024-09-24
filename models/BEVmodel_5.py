import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, \
    Flatten, Reshape, MaxPool2D, \
    Conv2DTranspose,  \
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


def conv_block(entrylayer, channels_out, kernelsize, stride, pad):
    c = Conv2D(channels_out,kernel_size=kernelsize, strides=stride,
               kernel_initializer = tf.keras.initializers.HeNormal(), 
               kernel_regularizer=l2(0.01),
               padding = pad, use_bias = False)(entrylayer)
    c = BatchNormalization()(c)
    c = Activation('relu')(c)
    return c
def depthwise_sep_block(entrylayer, channels, kernelsize, stride, pad='same'):
    # Depthwise Convolution
    c = DepthwiseConv2D(kernel_size=kernelsize, strides=stride, padding=pad, use_bias=False)(entrylayer)
    c = BatchNormalization()(c)
    c = Activation('relu')(c)

    # Pointwise Convolution
    c = Conv2D(filters=channels, kernel_size=(1, 1), use_bias=False,kernel_regularizer=l2(0.01),)(c)
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

    # last_class = Conv2D(32, (3,3), padding = 'same', name="class1", activation="relu")(entry_layer)
    last_class = Conv2D(nb_anchors, (1, 1))(entry_layer)
    last_class = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    # last_conf = Conv2D(128, (3,3), padding = 'same', name="occ1", activation="relu")(entry_layer)
    last_conf = Conv2D(nb_anchors, (1, 1), name="occ2", activation="sigmoid")(entry_layer)
    last_conf = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, 1), name="F_occ")(last_conf)

    # last_pos = Conv2D(128, (3,3), padding = 'same', name="loc1", activation="relu")(entry_layer)
    last_pos = Conv2D(nb_anchors*3, (1, 1), name="loc2")(entry_layer)
    last_pos =  Reshape(tuple(i//cfg.factor for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    # last_dim = Conv2D(128, (3,3), padding = 'same', name="size1", activation="relu")(entry_layer)
    last_dim = Conv2D(nb_anchors*3, (1, 1), name="size2")(entry_layer)
    last_dim = Reshape(tuple(i//cfg.factor for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    # last_rot = Conv2D(128, (3,3), padding = 'same', name="rot1", activation="relu")(entry_layer)
    last_rot = Conv2D(nb_anchors, (1, 1), name="rot2")(entry_layer)
    last_rot = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, 1), name="F_rot")(last_rot)

    output = Concatenate()([last_conf,last_pos,last_dim,last_rot,last_class])

    return output

def Img2BEV(img):
    C1 = conv_block(img, nb_channels, (2,2), (2,2), pad='same')
    C1_DW = conv_block(C1, C1.shape[-1], (1,1), (1,1), pad='valid')
    C1_DW = MaxPool2D((2,2))(C1_DW)

    C2 = conv_block(C1, nb_channels, (3,3), (2,2), pad='same')
    C2_out = Concatenate(axis=-1)([C1_DW,C2]) # 256
    C2_DW = conv_block(C2_out, C2_out.shape[-1], (1,1), (1,1), pad='valid')
    C2_DW = MaxPool2D((2,2))(C2_DW)

    C3 = conv_block(C2, nb_channels, (3,3), (2,2), pad='same')
    C3_out = Concatenate(axis=-1)([C2_DW,C3])
    C3_DW = conv_block(C3_out, C3_out.shape[-1], (1,1), (1,1), pad='valid')
    C3_DW = MaxPool2D((2,2))(C3_DW)

    C4 = conv_block(C3, nb_channels, (3,3), (2,2), pad='same')
    C4_out = Concatenate(axis=-1)([C3_DW,C4])
    C4_out = conv_block(C4_out, C4_out.shape[-1], (1,1), (1,1), pad='valid')

    # UP stage
    C2_up = conv_block_up(C2_out, nb_channels, (3,3), (2,2), pad = 'same')
    C3_up = conv_block_up(C3_out, nb_channels, (3,3), (4,4), pad = 'same')
    C4_up = conv_block_up(C4_out, nb_channels, (3,3), (8,8), pad = 'same')

    # Mix everything
    out = Concatenate()([C1,C2_up,C3_up,C4_up])
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