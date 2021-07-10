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

import tensorflow as tf
import argparse
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
import sklearn
import random
import os

parser = argparse.ArgumentParser(description='Monocular Depth Estimation - Evaluation')
parser.add_argument('--data', default="nyu", type=str, help='Data set')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')

args = parser.parse_args()

model = tf.keras.models.load_model('./models/model.h5')

def _parse_function(filename, label):
    # Read images from disk
    # shape_color = (240, 320, 3)
    shape_depth = (240, 320, 1) # <-- adjusted the image size to match [Alhashim and Wonka, 2019]

    image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
    depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)),
                                    [shape_depth[0], shape_depth[1]])

    # Format
    color = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)

    # Normalize the depth values (in cm)
    depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)
    return color, depth

if args.data == 'nyu':
    ## Test_dataset
    csv_file_test = './nyu-v2/data/nyu2_test.csv'
    csv_test = open(csv_file_test, 'r').read()
    nyu2_test = list((row.split(',') for row in (csv_test).split('\n') if len(row) > 0))
    nyu2_test = sklearn.utils.shuffle(nyu2_test, random_state=0)
    filenames_test = [os.path.join('./nyu-v2', i[0]) for i in nyu2_test]
    labels_test = [os.path.join('./nyu-v2', i[1]) for i in nyu2_test]
    length_test = len(filenames_test)
    dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test, labels_test))
    dataset_test = dataset_test.shuffle(buffer_size=len(filenames_test), reshuffle_each_iteration=True)
    dataset_test = dataset_test.repeat()
    dataset_test = dataset_test.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    batch_size = args.batch_size  # batch_size from inputs, default value is 2
    test_generator = dataset_test.batch(batch_size=batch_size)
else:
    raise NotImplementedError

###############################################################################
# Evaluate model - Metrics: RMSE, MAE
###############################################################################

model.compile(loss=None, metrics=[tf.keras.metrics.RootMeanSquaredError(),
                                  tf.keras.metrics.MeanAbsoluteError()])
# TODO: add additional metrics as required (see: https://keras.io/api/metrics/)
score = model.evaluate(test_generator, steps=length_test, verbose=1)
print("[RMSE, MAE] ", score[1:])
