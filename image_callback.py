import numpy as np
import tensorflow as tf

class ImageLoggingCallback(tf.keras.callbacks.Callback):
	def __init__(self, log_dir, validation_data, freq=5, X_div=None, Z_div=None):
		super().__init__()
		self.log_dir = log_dir
		self.validation_data = validation_data
		self.freq = freq
		self.X_div = X_div
		self.Z_div = Z_div
		self.writer = tf.summary.create_file_writer(log_dir + '/images')
		self.ground_truth_logged = False  # To control the logging of ground truth

	def on_epoch_end(self, epoch, logs=None):
		if epoch % self.freq == 0:  # Log every 'freq' epochs
			val_images, val_labels = next(iter(self.validation_data))
			dt = self.model.predict(val_images, batch_size=1)  # Predict the whole batch
			# print(val_images[0].shape)
			# exit()
			for i in range(len(val_images[0])):
				# Reshape and prepare the prediction image
				occupancy = np.reshape(dt[i, ..., 0], (self.X_div, self.Z_div, 2))
				img = np.concatenate((occupancy, np.zeros((self.X_div, self.Z_div, 1))), axis=-1)

				with self.writer.as_default():
					# Log predicted occupancy
					tf.summary.image("Epoch:{}_Prediction_{}".format(epoch, i), np.expand_dims(img, axis=0), step=epoch)

					# Log ground truth conf_matrix only once or based on a condition
					if not self.ground_truth_logged:
						for j in range(len(val_images[0])):
							conf_matrix = np.reshape(val_labels[j, ..., 0], (self.X_div, self.Z_div, 2))
							conf_matrix_img = np.concatenate((conf_matrix, np.zeros((self.X_div, self.Z_div, 1))), axis=-1)
							tf.summary.image("Ground_Truth_{}".format(j), np.expand_dims(conf_matrix_img, axis=0), step=0)
						self.ground_truth_logged = True  # Prevent further logging of ground truth

				self.writer.flush()