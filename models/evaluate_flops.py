import tensorflow as tf
from ConvNext import My_Model  # Assuming this is your model's structure as defined above
import numpy as np
from tensorflow.keras.layers import Input
# Define dummy inputs based on the input shapes your model expects
# input_pillar = Input(shape=(12000, 100, 9), name='input_pillar')
# input_pillar_mean = Input(shape=(12000, 3), name='input_pillar_mean')
input_img = Input(shape=(512, 512, 3), name='input_img')

# Construct the model
output = My_Model(input_img)
model = tf.keras.Model(inputs=[input_img], outputs=output)





def calculate_flops(model):
    total_flops = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Get layer parameters
            kernel_height, kernel_width = layer.kernel_size
            output_channels = layer.filters
            input_channels = layer.input_shape[-1]
            output_height, output_width = layer.output_shape[1:3]
            
            # Calculate FLOPs
            layer_flops = 2 * kernel_height * kernel_width * input_channels * output_channels * output_height * output_width
            if layer.use_bias:
                layer_flops += output_channels * output_height * output_width
            total_flops += layer_flops

        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            # Get layer parameters
            kernel_height, kernel_width = layer.kernel_size
            input_channels = layer.input_shape[-1]
            output_height, output_width = layer.output_shape[1:3]
            
            # Calculate FLOPs
            layer_flops = kernel_height * kernel_width * input_channels * output_height * output_width
            if layer.use_bias:
                layer_flops += input_channels * output_height * output_width
            total_flops += layer_flops

        # Add other layer types as needed...

    return total_flops / 1e9  # Convert to GFLOPs

# Assuming `model` is your loaded model
gflops = calculate_flops(model)
print(f"Estimated Total GFLOPs: {gflops}")