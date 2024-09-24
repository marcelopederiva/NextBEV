import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, \
    Flatten, Reshape, MaxPool2D, \
    Conv2DTranspose,  \
    Add, Concatenate, Activation, Softmax, LayerNormalization
import numpy as np
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import activations
from tensorflow.keras.applications import ResNet50, MobileNetV2, InceptionV3
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras import activations

import config_model as cfg
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import add, Activation
from tensorflow.keras.regularizers import l2

# max_group = cfg.max_group
# max_pillars = cfg.max_pillars
nb_channels = cfg.nb_channels
batch_size = cfg.BATCH_SIZE
image_size = cfg.img_shape
nb_anchors = cfg.nb_anchors
nb_classes = cfg.nb_classes

def SmallBlock(entrylayer, channels, kernelsize, stride, pad='same', activation='relu', number='1'):
    # Depthwise Convolution
    c = DepthwiseConv2D(kernel_size=kernelsize, strides=stride, padding=pad, use_bias=False, name='dw_conv' + number)(entrylayer)
    c = BatchNormalization(fused=True, name='BN_dw' + number)(c)
    c = Activation(tf.keras.activations.relu)(c)

    # Pointwise Convolution
    c = Conv2D(filters=channels, kernel_size=(1, 1), use_bias=False, name='pw_conv' + number)(c)
    c = BatchNormalization(fused=True, name='BN_pw' + number)(c)
    c = Activation(tf.keras.activations.relu)(c)
    return c

def SmallBlock_up(entrylayer, channels, kernelsize, stride, pad = 'same', activation = 'relu', number='1'):
    c = Conv2DTranspose(filters = channels,
                        kernel_size = kernelsize,
                        strides = stride,
                        padding=pad,
                        kernel_initializer = tf.keras.initializers.HeNormal(),
                        name = 'convTransp'+number)(entrylayer)
    c = BatchNormalization(fused = True,name = 'BN'+ number)(c)
    c = Activation(tf.keras.activations.swish)(c)
    return c

def Head(entry_layer):

    last_class = Conv2D(nb_classes*nb_anchors, (1, 1))(entry_layer)
    last_class = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="F_clf")(last_class)

    last_conf = Conv2D(32, (3,3), padding = 'same', name="occ1", activation="linear")(entry_layer)
    last_conf = Conv2D(nb_anchors, (1, 1), name="occ2", activation="sigmoid")(last_conf)
    last_conf = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_occ")(last_conf)

    last_pos = Conv2D(32, (3,3), padding = 'same', name="loc1", activation="linear")(entry_layer)
    last_pos = Conv2D(nb_anchors*3, (1, 1), name="loc2")(last_pos)
    last_pos =  Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_pos")(last_pos)

    last_dim = Conv2D(32, (3,3), padding = 'same', name="size1", activation="linear")(entry_layer)
    last_dim = Conv2D(nb_anchors*3, (1, 1), name="size2")(last_dim)
    last_dim = Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="F_dim")(last_dim)

    last_rot = Conv2D(32, (3,3), padding = 'same', name="rot1", activation="linear")(entry_layer)
    last_rot = Conv2D(nb_anchors, (1, 1), name="rot2")(last_rot)
    last_rot = Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, 1), name="F_rot")(last_rot)



    output = Concatenate()([last_conf,last_pos,last_dim,last_rot,last_class])

    return output

def Img2BEV(img):
    # Input = 512x512x3

    # ----------------------------------------------------------------------------------------------------#

    def depthwise_separable_conv(x, filters, kernel_size, strides, name):
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', name=name+'_depthwise')(x)
        x = tf.keras.layers.BatchNormalization(name=name+'_bn1')(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same', name=name+'_pointwise')(x)
        x = tf.keras.layers.BatchNormalization(name=name+'_bn2')(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        return x

    # Apply depthwise separable convolutions
    c1 = depthwise_separable_conv(img, filters=188, kernel_size=(3, 3), strides=(1, 1), name='IMG_decay_1')
    Reshape_c1 = tf.keras.layers.Reshape((512, 188, 512), name='Out_c1')(c1)
    out_c1 = depthwise_separable_conv(Reshape_c1, filters=64, kernel_size=(3, 3), strides=(1, 1), name='IMG_decay_1_1')
    
    c2 = depthwise_separable_conv(c1, filters=126, kernel_size=(3, 3), strides=(2, 2), name='IMG_decay_2')
    Reshape_c2 = tf.keras.layers.Reshape((512, 126, 128), name='Out_c2')(c2)
    out_c2 = depthwise_separable_conv(Reshape_c2, filters=64, kernel_size=(3, 3), strides=(1, 1), name='IMG_decay_2_1')
    
    c3 = depthwise_separable_conv(c2, filters=72, kernel_size=(3, 3), strides=(2, 2), name='IMG_decay_3')
    Reshape_c3 = tf.keras.layers.Reshape((512, 72, 32), name='Out_c3')(c3)
    out_c3 = depthwise_separable_conv(Reshape_c3, filters=64, kernel_size=(3, 3), strides=(1, 1), name='IMG_decay_3_1')
    
    c4 = depthwise_separable_conv(c3, filters=66, kernel_size=(3, 3), strides=(2, 2), name='IMG_decay_4')
    Reshape_c4 = tf.keras.layers.Reshape((512, 66, 8), name='Out_c4')(c4)
    out_c4 = depthwise_separable_conv(Reshape_c4, filters=64, kernel_size=(3, 3), strides=(1, 1), name='IMG_decay_4_1')
    
    c5 = depthwise_separable_conv(c4, filters=240, kernel_size=(3, 3), strides=(2, 2), name='IMG_decay_5')
    Reshape_c5 = tf.keras.layers.Reshape((512, 60, 8), name='Out_c5')(c5)
    out_c5 = depthwise_separable_conv(Reshape_c5, filters=64, kernel_size=(3, 3), strides=(1, 1), name='IMG_decay_5_1')

    # Concatenate the outputs
    out = tf.keras.layers.Concatenate(axis=2)([out_c5, out_c4, out_c3, out_c2, out_c1])

    return out

def CrossAtt(Flidar,Fimg,number = '0'):
    # Reshape_Flidar = Reshape(((image_size[0]//4)*(image_size[0]//4),64), name ='Reshape_Flidar')(Flidar)
    # Reshape_Fimg = Reshape(((image_size[0]//4)*(image_size[0]//4),64), name ='Reshape_Fimg')(Fimg)

    # Self-Attention

    Query= Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/Query'+ number)(Flidar)
    Key= Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/Key'+ number)(Fimg)
    Value= Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/Value'+ number)(Fimg)

    out = tf.matmul(Query,tf.transpose(Key,perm=[0,2,3,1]))
    out = out/np.sqrt(image_size[0]//4)
    out = tf.keras.activations.softmax(out, axis = -1)
    out = tf.matmul(out,Value)
    out = Conv2D(64,(1,1), activation = 'linear', use_bias = False, name = 'SA/out'+ number)(out)
    # out = Reshape((128,128,64), name = 'Reshape_out_CrAtt')(out)
    
    return out


def My_Model(input_img):
    # Enter the pillar and pillar mean
    # Convolutional Network feature
    Fimg = Img2BEV(input_img)
    
    c_img1 = SmallBlock(Fimg, nb_channels,(3,3),(2,2),pad='same', number='_c_img')#256
    c_img2 = SmallBlock(c_img1, nb_channels,(3,3),(2,2),pad='same', number='_c_img2')#128
    c = SmallBlock_up(c_img2, nb_channels*2,(3,3),(2,2),pad='same', number='_up_1')#256
    # Head detection
    out = Head(c)
    return out


if __name__ == '__main__':
    from keras.layers import Input
    from keras.models import Model

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