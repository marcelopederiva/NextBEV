# Step 1: Import necessary libraries
import tensorflow as tf
import numpy as np
import time
from MobileNetv3large import My_Model  # Import the My_Model function from model.py
# from ResNet import My_Model
import config_model as cfg  # Import configuration settings

# Step 2: Define a function to create dummy input data
def generate_dummy_data():
    # input_pillar_shape = (cfg.BATCH_SIZE, 12000, 100, 9)
    # input_pillar_mean_shape = (cfg.BATCH_SIZE, 12000, 3)
    input_img_shape = (cfg.BATCH_SIZE, 512, 512, 3)

    # Generate dummy data
    # input_pillar = np.random.random(input_pillar_shape).astype(np.float32)
    # input_pillar_mean = np.random.random(input_pillar_mean_shape).astype(np.float32)
    input_img = np.random.random(input_img_shape).astype(np.float32)
    
    return  input_img

# Step 3: Function to evaluate model speed
def evaluate_model_speed():
    # Generate dummy data
    input_img = generate_dummy_data()

    # Create TensorFlow placeholders for inputs
    # input_pillar_ph = tf.keras.Input(shape=(12000, 100, 9), batch_size=cfg.BATCH_SIZE)
    # input_pillar_mean_ph = tf.keras.Input(shape=(12000, 3), batch_size=cfg.BATCH_SIZE)
    input_img_ph = tf.keras.Input(shape=(512, 512, 3), batch_size=cfg.BATCH_SIZE)

    # Load the model
    # output = My_Model(input_img_ph)
    
    
    output = My_Model(input_img_ph)
    # print(type(output))
    # exit()
    model = tf.keras.Model(inputs=[input_img_ph], outputs=output)
    model.summary()
    for i in range(20):
        # Measure model evaluation speed
        
        start_time = time.time()
        # Simulate a prediction
        model.predict([input_img])

        end_time = time.time()
        

        # model.summary()
        time_v = end_time - start_time

    print(f"Model evaluation took {1000*(time_v):.4f} ms.")
    
# Execute the model evaluation speed test
if __name__ == "__main__":
    evaluate_model_speed()
