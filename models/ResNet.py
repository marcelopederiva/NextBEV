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
from tensorflow.keras.applications import ResNet50
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

def My_Model(input_img):
    # Enter the pillar and pillar mean
    # Convolutional Network feature

    resnet = ResNet50(
        include_top=False,
        weights= None,
        input_tensor=None,
        pooling=None
    )
    Fimg = resnet(input_img)
    Fimg = conv_block_up(Fimg,1024,3,2) # 32x32
    Fimg = conv_block_up(Fimg,512,3,2) # 64x64
    Fimg = conv_block_up(Fimg,256,3,2) # 128x128
    Fimg = conv_block_up(Fimg,128,3,2) # 256x256
    # print(Fimg.shape)
    # exit()
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
    # model.summary()
    # tf.keras.utils.plot_model(
    # model,
    # to_file="D:/SCRIPTS/Doc_lidar_v2/models/model2.png",
    # show_shapes=True,
    # dpi = 300)