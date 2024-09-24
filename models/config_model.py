import numpy as np
''' **************  SHAPE  *****************  '''
# img_shape = (600,600)
# input_pillar_shape = (10000, 20, 9)
# input_pillar_indices_shape = (10000, 3)
''' **************  SHAPE  *****************  '''
# input_pillar_shape = (12000, 100, 9)
# input_pillar_indices_shape = (12000, 3)
img_shape = (512,512)
factor = 2
''' **************  DIVISIONS  *****************  '''
# X_div = 256
# Y_div = 1
# Z_div = 256

X_div = img_shape[0]//factor
Y_div = 1
Z_div = img_shape[1]//factor

# X_div = 500
# Y_div = 1
# Z_div = 500

''' **************  PILLAR  *****************  '''
# max_group = 20
# max_pillars = 10000
# nb_channels = 64

''' **************  PILLAR  *****************  '''
# max_group = 100
# max_pillars = 12000
nb_channels = 128
# nb_anchors = 4
nb_anchors = 2
# nb_classes = 4
# classes = {"Car":               0,
#                "Pedestrian":        1,
#                "Person_sitting":    1,
#                "Cyclist":           2,
#                "Truck":             3,
#                "Van":               3,
#                "Tram":              3,
#                "Misc":              3,
#                }

nb_classes = 1

classes = {"Car":               0
               }

# # width,height,lenght,orientation
# anchor = np.array([ [1.6,1.56,3.9,-1, 0],
#                     [1.6,1.56,3.9,-1, 1.57],
#                     [0.6,1.73,0.8,-0.6, 0],
#                     [0.6,1.73,0.8,-0.6, 1.57]], dtype=np.float32).tolist()

anchor = np.array([ [1.6,1.56,3.9,-1, 0],
                    [1.6,1.56,3.9,-1, 1.57]], dtype=np.float32).tolist()


trust_treshold = 0.5
pos_iou = 0.6
neg_iou = 0.4

color_true = { "0": 'limegreen',
                "1": 'cyan',
                "2": 'blue',
               "3": 'pink',
               }


''' **************  PARAMETERS  *****************  '''
BATCH_SIZE = 1
ITERS_TO_DECAY = 100980 # 15*4*ceil(6788/4)  --> every 15 epochs on 6788 samples
LEARNING_RATE = 2e-4
DECAY_RATE = 1e-8

''' **************  PATHS  *****************  '''
# DATASET_PATH = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/'
DATASET_PATH = 'C:/Users/maped/Documents/Scripts/KITTI/data/'
TRAIN_TXT = 'train_80_car.txt'
VAL_TXT = 'val_20_car.txt'
VAL_MINI_TXT = 'val_mini_car.txt'
# TRAIN_TXT = 'train_80.txt'
# VAL_TXT = 'val_20.txt'

# LABEL_PATH = 'D:/SCRIPTS/Doc_code/data/label_2/'
# LIDAR_PATH = 'D:/SCRIPTS/Doc_code/data/input/lidar_crop_mini/'
# LABEL_PATH = '/home/rtxadmin/Documents/Marcelo/KITTI/label_2/'
# LIDAR_PATH = '/home/rtxadmin/Documents/Marcelo/Doc_code/data/input/lidar_crop/'
# IMG_PATH = '/home/rtxadmin/Documents/Marcelo/Img_datasets/KITTI_img/'
KITTI_PATH = 'C:/Users/maped/Documents/Scripts/KITTI/object/training/'
LABEL_PATH = 'C:/Users/maped/Documents/Scripts/KITTI/object/training/label_2/'

LIDAR_PATH =  'C:/Users/maped/Documents/Scripts/KITTI/data/input/lidar_crop/'
IMG_PATH = 'C:/Users/maped/Documents/Scripts/KITTI/object/training/image_2/'
''' **************  LAMBDA WEIGHTS  *****************  '''

######################### Pillar loss
# lambda_class = 0.5
# lambda_occ = 3.0
# lambda_pos = 2.0
# lambda_dim = 2.0
# lambda_rot = 1.0

######################## Pillar loss
lambda_class = 0.2
lambda_occ = 1.0
lambda_pos = 2.0
lambda_dim = 2.0
lambda_rot = 2.0

# ######################### Pillar loss 2
# lambda_class = 0.0
# lambda_occ = 0.5
# lambda_pos = 2.0
# lambda_dim = 2.0
# lambda_rot = 1.0

''' **************  NORMALIZATION  *****************  '''
#Norm Final -- KITTI ---- Best Until the moment

x_min = -40
x_max = 40
x_diff = abs(x_max - x_min)

y_min = -2
y_max = 6
y_diff = abs(y_max - y_min)

z_min = 0
z_max = 80
z_diff = abs(z_max - z_min)

# w_max = 3
# h_max = 4.5
# l_max = 10

rot_norm = 3.1416 #direction doesent matter

stepx = x_diff/img_shape[0]
stepz = z_diff/img_shape[1]