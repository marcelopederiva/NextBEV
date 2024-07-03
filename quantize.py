import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from models.BEVmodel_9 import My_Model
import config_model as cfg

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


weights_path = 'checkpoints/val_loss/Temp_loss/model_030_Model_minimum.hdf5'
###############################-

np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})

# input_pillar = Input(input_pillar_shape,batch_size = 1)
# input_pillar_indices = Input(input_pillar_indices_shape,batch_size = 1)
input_img = Input((cfg.img_shape[0],cfg.img_shape[1],3), batch_size = 1)

output = My_Model(input_img)
model = Model(inputs=[input_img], outputs=output)
model.load_weights(weights_path, by_name=True)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization flag to use default optimization strategies, which includes quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
# Convert the model
tflite_quantized_model = converter.convert()

# Save the quantized model to a file
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quantized_model)