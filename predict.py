###############################################################################
# SLAM Winter School 2021
# Project #9 - Monocular Depth Estimation via Transfer Learning
#
# Team 9b: Javed Ahmad
#          Yatharth Ahuja
#          Krunal Chande
#          Thomas Pintaric
#          Veeresh Taranalli
#
###############################################################################

import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser(description='Monocular Depth Estimation - Prediction')
parser.add_argument('--color_input', type=str, default='color.png', help='Color image (input)')
parser.add_argument('--depth_output', type=str, default='depth.png', help='Predicted depth image (output)')
parser.add_argument('--visualize_result', default=False, action='store_true', help='Visualize output side-by-side with input')
parser.add_argument('--max_depth', type=int, default=1000, help='Maximal depth value')
args = parser.parse_args()

model = tf.keras.models.load_model('./models/model.h5')

###############################################################################
# Predict depth from RGB
###############################################################################
color = tf.image.decode_jpeg(tf.io.read_file(args.color_input))
predicted_depth = model.predict(np.expand_dims(tf.image.convert_image_dtype(color, dtype=tf.float32), axis=0))

###############################################################################
# Save result (predicted depth image)
###############################################################################

normalized_depth_prediction = preprocessing.minmax_scale(predicted_depth[0, :, :, 0])

# Save as grayscale
# Image.fromarray(np.uint8(normalized_depth_prediction * 255)).save(args.depth_output)

# Apply colormap before saving
colormap = plt.get_cmap('viridis')
Image.fromarray(np.uint8(colormap(normalized_depth_prediction)*255)).convert('RGB').save(args.depth_output)

###############################################################################
# Visualize result (optional)
###############################################################################
if args.visualize_result:
    plt.subplot(1, 2, 1)
    plt.imshow(color)
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_depth[0, :, :, 0])
    plt.show()
