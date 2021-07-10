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

parser = argparse.ArgumentParser(description='Monocular Depth Estimation - Training')
parser.add_argument('--max_images', type=int, default=0, help='Max. number of images used for training (0 = use all available images')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size') # NOTE: in the paper, the authors use a batch size of 8
parser.add_argument('--max_depth', type=int, default=1000, help='Maximal depth value')
parser.add_argument('--data', default="nyu", type=str, help='Name of the training data set')
parser.add_argument('--use_naive_L1_loss', action='store_true', default=False, help='Use naive point-wise L1 loss based on the depth values')
parser.add_argument('--train_on_multiple_gpus', action='store_true', default=False, help='Use multiple GPUs for training')
parser.add_argument('--apply_data_augmentations', action='store_true', default=False, help='Apply data augmentations')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')

args = parser.parse_args()
print("Available physical devices (GPUs): ", len(tf.config.list_physical_devices('GPU')))

if args.use_naive_L1_loss:
    print("Using naive point-wise L1 loss based on the depth values.")

if args.train_on_multiple_gpus:
    print("Using multiple GPUs for training, if available.")

if args.apply_data_augmentations:
    print("Data augmentations will be applied.")

###############################################################################
# NYU-V2 dataset
###############################################################################

def channelSwap(color):
    channel_order = random.sample(range(3), 3)
    color_channels = tf.unstack(color, num=3, axis=-1)
    ch0, ch1, ch2 = color_channels[channel_order[0]],\
                    color_channels[channel_order[1]],\
                    color_channels[channel_order[2]]
    return tf.stack([ch0, ch1, ch2], axis=-1)

def gaussianNoise(color):
    noise = tf.random.normal(shape=tf.shape(color), 
                            mean=0.0, 
                            stddev=(random.randint(1, 10)/100), 
                            dtype=tf.float32)
    return color + noise

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

    if args.apply_data_augmentations:
        # Apply data augmentations
        # See: https://www.tensorflow.org/hub/tutorials/cropnet_on_device

        color = tf.image.random_brightness(color, 0.2)
        color = tf.image.random_contrast(color, 0.5, 2.0)
        color = tf.image.random_saturation(color, 0.75, 1.25)
        color = tf.image.random_hue(color, 0.1)
        
        if random.randint(1, 4) == 1:
            color = channelSwap(color)
            
        if random.randint(1, 4) == 1:
            color = gaussianNoise(color)

        if bool(random.getrandbits(1)):
            color = tf.image.flip_left_right(color)
            depth = tf.image.flip_left_right(depth)

    return color, depth

if args.data == 'nyu':

    # ----------------------------------------
    # Training dataset
    # ----------------------------------------

    csv_file = './nyu-v2/data/nyu2_train.csv'
    csv = open(csv_file, 'r').read()
    nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))
    nyu2_train = sklearn.utils.shuffle(nyu2_train, random_state=0)

    # optionally train on fewer images (only for debugging purposes)
    if (args.max_images > 0) and (args.max_images < len(nyu2_train)):
        del nyu2_train[args.max_images:-1]

    filenames = [os.path.join('./nyu-v2', i[0]) for i in nyu2_train]
    labels = [os.path.join('./nyu-v2', i[1]) for i in nyu2_train]

    length = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=len(filenames), reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.map(map_func=_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    batch_size = args.batch_size  # batch_size from inputs, default value is 2
    train_generator = dataset.batch(batch_size=batch_size)

    # ----------------------------------------
    # Test dataset
    # ----------------------------------------

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

###############################################################################
# Encoder/decoder network
###############################################################################

def create_model():
    dense169 = DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet')
    dense169.trainable = True
    dense169_output = dense169.layers[-1].output
    number_of_filters = int(dense169_output.shape[-1])

    output = Conv2D(filters=number_of_filters, kernel_size=1, padding='same', name='CONV2')(dense169_output)

    for layer_name in ['pool3_pool', 'pool2_pool', 'pool1', 'conv1/relu']:
        output = UpSampling2D(size=(2, 2), interpolation='bilinear')(output)
        output = Concatenate()([output, dense169.get_layer(layer_name).output])
        number_of_filters = number_of_filters // 2
        output = Conv2D(filters=number_of_filters, kernel_size=3, strides=1, padding='same')(output)
        output = LeakyReLU(alpha=0.2)(output)
        output = Conv2D(filters=number_of_filters, kernel_size=3, strides=1, padding='same')(output)
        output = LeakyReLU(alpha=0.2)(output)

    output = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(output)
    return tf.keras.Model(inputs=dense169.inputs, outputs=output)

###############################################################################
# Loss functions
###############################################################################

# -----------------------------------------------------------------------------
# "Naive" loss function: point-wise L1 loss defined on the depth values

def naive_depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    l_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)
    return tf.keras.backend.mean(l_depth)

# -----------------------------------------------------------------------------
# Loss function as given by [Alhashim and Wonka, 2019]

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):

    # The first loss term L_depth is the point-wise L1 loss defined on the depth values.
    # See: [Alhashim and Wonka, 2019], Section 3.2, Eq. (2)
    L_depth = tf.keras.backend.mean(tf.keras.backend.abs(y_pred - y_true), axis=-1)

    # The second loss term L_grad is the L1 loss defined over the image gradient of the depth image.
    # See: [Alhashim and Wonka, 2019], Section 3.2, Eq. (3)
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    L_grad = tf.keras.backend.mean(tf.keras.backend.abs(dy_pred - dy_true) +
                                   tf.keras.backend.abs(dx_pred - dx_true), axis=-1)

    # The third loss term L_ssim uses the mean Structural Similarity (mSSIM).
    # See: [Alhashim and Wonka, 2019], Section 3.2, Eq. (4)

    # NOTE: The authors don't mention the size/width of the gaussian filter and/or the values of the parameters k1/k2.

    L_ssim = tf.keras.backend.mean(tf.keras.backend.clip(((1.0 - tf.image.ssim(
        img1=y_true, img2=y_pred, max_val=maxDepthVal, filter_size=5, filter_sigma=1.0)) / 2.0), min_value=0.0, max_value=1.0))

    # Final loss function L is weighted sum of L_depth, L_grad and L_ssim
    # See: [Alhashim and Wonka, 2019], Section 3.2, Eq. (1)

    # NOTE: Our assumption is that the parameter "theta" is equivalent to "lambda" from the paper,
    #       which the authors empirically set to 0.1
    return (theta * L_depth) + L_grad + L_ssim

###############################################################################
# Training (optionally on multiple GPUs)
###############################################################################

if args.train_on_multiple_gpus:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices (GPUs) used for training: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = create_model()
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
                      loss=naive_depth_loss_function if args.use_naive_L1_loss else depth_loss_function,
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
else:
    model = create_model()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001, amsgrad=True),
                  loss=naive_depth_loss_function if args.use_naive_L1_loss else depth_loss_function,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

model.fit(train_generator, epochs=args.epochs, steps_per_epoch=length//batch_size)

###############################################################################
# Save trained model
###############################################################################

model.save("./models/model.h5", include_optimizer=False)
model.save('./models/', save_format='tf', include_optimizer=False)

###############################################################################
# Evaluate trained model
###############################################################################

score = model.evaluate(test_generator, steps=length_test, verbose=1)
print("[Loss, RMSE]: {}".format(score))
