import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, \
    Flatten, Reshape, MaxPool2D, UpSampling2D,Conv2DTranspose,\
    Add, Concatenate, Activation, Softmax, LayerNormalization
from groupnorm import GroupNormalization
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

def ws_reg(kernel):
  kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
  kernel = kernel - kernel_mean
  kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
  kernel = kernel / (kernel_std + 1e-5)
#   return kernel

def SimpleConv2D(entry_layer, out_features, Ksize, stride=1, group=1, pad='same'):
    c = Conv2D(out_features,
               kernel_size=(Ksize, Ksize),
               strides=(stride, stride),
               groups=group,
               kernel_initializer = tf.keras.initializers.HeNormal(), 
               kernel_regularizer=ws_reg,
               padding=pad)(entry_layer)
    return c


def BottleNeckBlock(entry_layer, in_features, out_features, exp, stride=1, Ksize = 5):
    expanded_features = out_features * exp

    conv_seq = DepthwiseConv2D(kernel_size=Ksize,strides = stride, use_bias = False, padding = 'same')(entry_layer)

    conv_seq = GroupNormalization(axis=-1,groups=conv_seq.shape[-1])(conv_seq)

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
               kernel_regularizer=ws_reg,
               padding = pad, use_bias = False)(entrylayer)
    c = GroupNormalization(axis=-1, groups=c.shape[-1])(c)
    c = Activation('relu')(c)
    return c
def conv_block_up(entrylayer, channels, kernelsize, stride, pad = 'same'):
    c = Conv2DTranspose(filters = channels,
                        kernel_size = kernelsize,
                        strides = stride,
                        padding=pad,
                        kernel_regularizer=ws_reg,
                        kernel_initializer = tf.keras.initializers.HeNormal())(entrylayer)
    c = GroupNormalization(axis=-1, groups=c.shape[-1])(c)
    c = Activation('relu')(c)
    return c

def Head(entry_layer):

    last_class = Conv2D(nb_anchors, (1, 1))(entry_layer)
    last_class = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    last_conf = Conv2D(64, (1, 1), padding = 'valid', name="occ1", kernel_regularizer=ws_reg, kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_conf = GroupNormalization(axis=-1,groups=last_conf.shape[-1])(last_conf)
    last_conf = Activation('relu')(last_conf)
    last_conf = Conv2D(nb_anchors, (1, 1), name="occ2", activation="sigmoid")(last_conf)
    last_conf = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, 1), name="F_occ")(last_conf)

    last_pos = Conv2D(64, (1, 1), padding = 'valid', name="loc1", kernel_regularizer=ws_reg, kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_pos = GroupNormalization(axis=-1, groups=last_pos.shape[-1])(last_pos)
    last_pos = Activation('relu')(last_pos)
    last_pos = Conv2D(nb_anchors*3, (1, 1), name="loc2")(last_pos)
    last_pos =  Reshape(tuple(i//cfg.factor for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    last_dim = Conv2D(64, (1, 1), padding = 'valid', name="size1", kernel_regularizer=ws_reg, kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    last_dim = GroupNormalization(axis=-1, groups= last_dim.shape[-1])(last_dim)
    last_dim = Activation('relu')(last_dim)
    last_dim = Conv2D(nb_anchors*3, (1, 1), name="size2")(last_dim)
    last_dim = Reshape(tuple(i//cfg.factor for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    last_rot = Conv2D(16, (1, 1), padding = 'valid', name="rot1", kernel_regularizer='l2', kernel_initializer = tf.keras.initializers.HeNormal())(entry_layer)
    # last_rot = GroupNormalization(axis=-1, groups= last_rot.shape[-1])(last_rot)
    last_rot = Activation('relu')(last_rot)
    last_rot = Conv2D(nb_anchors, (1, 1), name="rot2")(last_rot)
    last_rot = Reshape(tuple(i // cfg.factor for i in image_size) + (nb_anchors, 1), name="F_rot")(last_rot)

    output = Concatenate()([last_conf,last_pos,last_dim,last_rot,last_class])

    return output

def Img2BEV(img):
    c1 = SimpleConv2D(img, out_features=16,
                     Ksize=3, stride=2, group=1,  pad='same')
    # c1 = BottleNeckBlock(c1, 64, 64, 4, stride=1)
    # for i in range(1):
    #     c1 = BottleNeckBlock(c1, 32, 32, 2, stride=1)
    c1 = GroupNormalization(axis=-1, groups=c1.shape[-1])(c1)
    c1 = Activation('relu')(c1) # 256 x 256 x 16

    
    # c2 = LayerNormalization()(c1)
    
    c2 = SimpleConv2D(c1, out_features=24,
                     Ksize=3, stride=2, group=1, pad='same')
    c2 = GroupNormalization(axis=-1, groups=c2.shape[-1])(c2)
    c2 = Activation('relu')(c2) # 128 x 128 x 24

    c2 = BottleNeckBlock(c2, 24, 24, exp=4, stride=1,Ksize=3)

    

    c3 = LayerNormalization()(c2)
    c3 = SimpleConv2D(c3, out_features=40,
                     Ksize=3, stride=2, group=1, pad='same')
    c3 = GroupNormalization(axis=-1, groups=c3.shape[-1])(c3)
    c3 = Activation('relu')(c3) # 64 x 64 x 40
    # c3 = BottleNeckBlock(c3, 128, 128, 4, stride=1)

    for i in range(3):
        c3 = BottleNeckBlock(c3, 40, 40, 4, stride=1)


    c4 = LayerNormalization(axis=-1)(c3)
    c4 = SimpleConv2D(c4, out_features=48,
                     Ksize=3, stride=1, group=1, pad='same')
    # c4 = BottleNeckBlock(c4, 256, 256, 4, stride=1)
    c4 = GroupNormalization(axis=-1, groups=c4.shape[-1])(c4)
    c4 = Activation('relu')(c4) # 64 x 64 x 48
    for i in range(3):
        c4 = BottleNeckBlock(c4, 48, 48, 4, stride=1)


    c5 = LayerNormalization()(c4)
    c5 = SimpleConv2D(c5, out_features=96,
                     Ksize=3, stride=2, group=1, pad='same')
    c5 = GroupNormalization(axis=-1, groups=c5.shape[-1])(c5)
    c5 = Activation('relu')(c5) # 32 x 32 x 96

    # c5 = BottleNeckBlock(c5, 256, 256, 4, stride=1)

    for i in range(3):
        c5 = BottleNeckBlock(c5, 96, 96, 4, stride=1)

    # UP step
    # print(c1.shape)  
    # print(c2.shape)  
    # print(c3.shape)  
    # print(c4.shape)  
    # print(c5.shape)  
    
    c_up1 = conv_block_up(c5,96,(3,3),(2,2)) # 64x64
    conc1 = Concatenate()([c_up1,c4,c3]) # 64x64x96+48+40 = 184
    c_up2 = conv_block_up(conc1,conc1.shape[-1],(3,3),(2,2)) # 128x128
    conc2 = Concatenate()([c_up2,c2]) # 128x128x184+24 = 208
    c_up3 = conv_block_up(conc2,conc2.shape[-1],(3,3),(2,2)) # 256x256
    conc3 = Concatenate()([c_up3,c1]) # 128x128x184+24 = 208

    # C2 - 256 // C3 - 128 // C4 - 64 // C5 - 64
    
    # c_5_4 = Concatenate()([c4,c5]) # 64.64.512c
    # c_54 = conv_block_up(c_5_4, 128, (3,3), (2,2)) # 128.128.256
    # c3_54 = Concatenate()([c3,c_54])# 128.128.384c
    # c_543 = conv_block_up(c3_54, 128, (3,3), (2,2)) # 256.256.256
    # c2_543 = Concatenate()([c2,c_543])# 256.256.384c

    # out = LayerNormalization()(c2_543)
    # out = SimpleConv2D(out, out_features=256,
    #                  Ksize=3, stride=1, group=1, pad='same')
    # out = BottleNeckBlock(out, 256, 256, 2, stride=1)
    # print(out.shape)
    # exit()
    
    return conc3

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