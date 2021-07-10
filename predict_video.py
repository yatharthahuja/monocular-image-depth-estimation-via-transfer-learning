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
import av

parser = argparse.ArgumentParser(description='Monocular Depth Estimation - Prediction')
parser.add_argument('--video_input', type=str, default='input.mp4', help='Color video input')
parser.add_argument('--video_output', type=str, default='output.mp4', help='Side-by side color/predicted depth video output')
args = parser.parse_args()

model = tf.keras.models.load_model('./models/model.h5')

color_input_container = av.open(args.video_input)
input_video_stream = color_input_container.streams.video[0]

fps = input_video_stream.guessed_rate

output_container = av.open(args.video_output, mode='w')
output_stream = output_container.add_stream('mpeg4', rate=fps)
output_stream.width = 2*640
output_stream.height = 480
output_stream.pix_fmt = 'yuv420p'
output_stream.bit_rate = 8000000

colormap = plt.get_cmap('viridis')

for frame in color_input_container.decode(video=0):

    if frame.index % 100 == 0:
        print('Processing frame {}'.format(frame.index))

    color_image = frame.to_image()

    if (color_image.height != 1080) or (color_image.width != 1920):
        print("Input video must be 1920x1080.")
        raise NotImplementedError

    box = (240, 0, 1920-240, 1080)
    color_image = color_image.crop(box)
    color_image = color_image.resize((640, 480))

    color_array = np.array(color_image)

    predicted_depth = model.predict(np.expand_dims(tf.image.convert_image_dtype(color_array, dtype=tf.float32), axis=0))
    normalized_depth_prediction = preprocessing.minmax_scale(predicted_depth[0, :, :, 0])

    colorized_depth = Image.fromarray(np.uint8(colormap(normalized_depth_prediction)*255)).convert('RGB')
    colorized_depth = colorized_depth.resize((640, 480))

    side_by_side = Image.new('RGB', (2*640, 480))
    side_by_side.paste(color_image, (0, 0))
    side_by_side.paste(colorized_depth, (640, 0))

    side_by_side_frame = av.VideoFrame.from_ndarray(np.array(side_by_side), format='rgb24')
    for packet in output_stream.encode(side_by_side_frame):
        output_container.mux(packet)

for packet in output_stream.encode():
    output_stream.mux(packet)

output_container.close()
