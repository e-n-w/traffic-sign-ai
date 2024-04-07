from keras import utils, layers, preprocessing
from keras_cv import layers as layers_cv
import tensorflow as tf
import numpy as np
import os
import random
from PIL import ImageFilter

us_data_path = "./us_data"
us_base_path = f"{us_data_path}/Base"
us_train_path = f"{us_data_path}/Train"

IMG_COUNT = 250
color_jitter = layers_cv.RandomColorJitter(brightness_factor=(-0.6, 0.3), contrast_factor=(0, 0.2), saturation_factor=(0.3, 0.6), hue_factor=0.015, value_range=(0,255))
rotation = layers.RandomRotation(factor=(-0.04, 0.03), fill_mode="constant", fill_value=0.0)
zoom = layers.RandomZoom(height_factor=(0.3, 0.5), fill_mode="constant", fill_value=0.0)
shear = layers_cv.RandomShear(x_factor=0.1, fill_mode="constant", fill_value=0.0)

for base_img_file in os.listdir(us_base_path):
    base_img_path = f"{us_base_path}/{base_img_file}"
    classid = base_img_file.split('.')[0]
    index = 0
    save_path = f"{us_train_path}/{classid}"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    img = preprocessing.image.load_img(base_img_path)
    img = tf.image.resize_with_pad(img, 128, 128)
    for index in range(IMG_COUNT + 1):
        img_arr = preprocessing.image.img_to_array(img)
        img_arr = zoom(img_arr)
        img_arr = shear(img_arr)
        img_arr = rotation(img_arr)
        img_arr = color_jitter(img_arr, training=True)
        rand_kernel = random.randint(0,4)
        img_blur = preprocessing.image.array_to_img(img_arr).filter(ImageFilter.GaussianBlur((rand_kernel, max(rand_kernel - 1, 0))))
        img_arr = preprocessing.image.img_to_array(img_blur)
        rand_height = random.randint(24, 48)
        rand_width = rand_height + random.randint(-4, 0)
        img_arr = layers.Resizing(rand_height, rand_width, interpolation="lanczos3")(img_arr)
        preprocessing.image.save_img(f"{save_path}/{classid}_{index}.png", img_arr)
