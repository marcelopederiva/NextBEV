import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import datetime
from models.ConvNext import My_Model
# from models.MobileNet import My_Model
from dataset import SequenceData
# from Pillar_loss_M import PointPillarNetworkLoss
from Pillar_loss_M_old import PointPillarNetworkLoss, focal_loss, loc_loss, size_loss, angle_loss, class_loss
# import numpy as np
import config_model as cfg
from accuracy import correct_grid, incorrect_grid
from image_callback import ImageLoggingCallback
EPOCHS = 180
BATCH_SIZE = cfg.BATCH_SIZE
ITERS_TO_DECAY = cfg.ITERS_TO_DECAY
LEARNING_RATE = cfg.LEARNING_RATE
DECAY_RATE = cfg.DECAY_RATE

# initial_learning_rate = 0.0002

initial_learning_rate = 0.001
factor = 0.6
patience = 3
min_lr = 1e-8

# input_pillar_shape = cfg.input_pillar_shape
# input_pillar_indices_shape = cfg.input_pillar_indices_shape
input_img_shape = cfg.img_shape
img_shape = cfg.img_shape

tf.get_logger().setLevel("ERROR")


DATASET_PATH = cfg.DATASET_PATH
LABEL_PATH = cfg.LABEL_PATH

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        tf.summary.scalar('learning rate', data=lr, step=epoch)

def train():
	batch_size = BATCH_SIZE

	# input_pillar = Input(input_pillar_shape, batch_size = batch_size)
	# input_pillar_indices = Input(input_pillar_indices_shape, batch_size = batch_size)
	input_img = Input((img_shape[0],img_shape[1],3),batch_size = batch_size)

	output = My_Model(input_img)
	model = Model(inputs=[input_img], outputs=output)
	# model.load_weights(os.path.join('checkpoints/val_loss/Temp_loss/', "model_010_Model_minimum.hdf5"))
	# print('Model Loaded!\n')
	#########################################
	#										#
	#               COMPILE                 #
	#									    #
	#########################################

	# loss = PointPillarNetworkLoss()

	optimizer = Adam(learning_rate = initial_learning_rate, clipnorm=1.0)
	
	# optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_learning_rate, 
	# 										clipnorm=1.0,
	# 										weight_decay=0.01,
	# 										beta_1=0.95,
	# 										beta_2=0.99)

	model.compile(optimizer = optimizer, loss=PointPillarNetworkLoss, metrics =[ correct_grid, incorrect_grid ,
																				 focal_loss, loc_loss, size_loss,
																				 angle_loss])
	# model.compile(optimizer = optimizer, loss=loss.losses())

	#########################################
	#										#
	#             CHECKPOINTS               #
	#                LOSS                   #
	#									    #
	#########################################

	save_dir_l = 'checkpoints/val_loss/'
	weights_path_l = os.path.join(save_dir_l,(datetime.datetime.now().strftime("%Y%m%d-") +'ConvNext_minimum.hdf5'))
	checkpoint_loss = ModelCheckpoint(weights_path_l, monitor = 'val_loss',mode='min', save_best_only = True)



	#########################################
	#										#
	#              CALLBACKS                # 
	#									    #
	#########################################

	# early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 1, mode = 'auto')

	log_dir = 'logs/'+ datetime.datetime.now().strftime("%Y%m%d-") + 'ConvNext'
	tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_grads = True)

	
	#---------------------------------------------------------------------------------------	
	#########################################
	#										#
	#               DATASET                 # 
	#									    #
	#########################################

	dataset_path = DATASET_PATH
	''' Insert SequenceData '''
	train_gen = SequenceData('train', dataset_path, img_shape, batch_size,data_aug = False)
	# print(len(train_gen))
	# exit()

	valid_gen = SequenceData('val', dataset_path, img_shape, batch_size,data_aug = False)

	valid_min_gen = SequenceData('val_min', dataset_path, img_shape, batch_size,data_aug = False)
	#########################################
	#										#
	#             CHECKPOINTS               #
	#                LOSS epoc              #
	#									    #
	#########################################checkpoint = keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5', period=5) 

	image_logger = ImageLoggingCallback(log_dir=log_dir, validation_data=valid_min_gen, freq=5, X_div=cfg.X_div, Z_div=cfg.Z_div)


	checkpoint_loss_e = ModelCheckpoint('checkpoints/val_loss/Temp_loss/model_{epoch:03d}_ConvNext_minimum.hdf5',
										save_weights_only = True, 
										save_freq = int(10*len(train_gen)))


	#---------------------------------------------------------------------------------------	
	#########################################
	#										#
	#              LR SCHEDULE              # 
	#									    #
	########################################

	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
												mode='min',
												 factor=factor,
												 patience=patience, 
												 min_lr=min_lr,
												 verbose=1)
	
	# def scheduler(epoch, lr):
	# 	if epoch % 10 == 0 and epoch != 0:
	# 		lr = lr*0.8
	# 		return lr
	# 	else:
	# 		return lr

	# lr_schd = tf.keras.callbacks.LearningRateScheduler(scheduler)
	#---------------------------------------------------------------------------------------	
	#########################################
	#										#
	#        MODEL FIT GENERATOR            # 
	#									    #
	########################################
	# model.summary()
	# exit()

	# label_files = os.listdir(LABEL_PATH)
	# epoch_to_decay = int(
 #        np.round(ITERS_TO_DECAY / BATCH_SIZE * int(np.ceil(float(len(label_files)) / BATCH_SIZE))))
	callbacks=[
				tbCallBack,
				checkpoint_loss,
				reduce_lr,
				checkpoint_loss_e,
				image_logger,
                LearningRateLogger()
					  ]
	# try:
	model.fit(
		train_gen,
		epochs = EPOCHS,
		validation_data=valid_gen,
		# steps_per_epoch=len(train_gen),
		callbacks=callbacks,
		# initial_epoch=10,
		# use_multiprocessing = True,
		# workers = 2
		)
	# except KeyboardInterrupt:
	# 	model.save('checkpoints/interrupt/Interrupt_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5')
	# 	print('Interrupt. Output saved')

	# try:
	# 	model.fit(
	# 		train_gen,
	# 		epochs = EPOCHS,
	# 		validation_data=valid_gen,
	# 		steps_per_epoch=len(train_gen),
	# 		callbacks=callbacks,
	# 		use_multiprocessing = True,
	# 		workers = 4
	# 		)
	# except KeyboardInterrupt:
	model.save('FINAL-'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.hdf5')
	# 	print('Interrupt. Output saved')

if __name__=='__main__':
	train()