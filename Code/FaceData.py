import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

batch_size = 32
img_height = 128
img_width = 128

data_dir = '/home/vivek/Downloads/archive/img_align_celeba'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width))
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# print(len(train_ds))
# for i in range(len(train_ds)):
# 	im,_ = train_ds[i]
# 	print(im)
# 	break

img,_ = next(iter(train_ds))
print(img)

print(15*"-")
img,_ = next(iter(train_ds))
print(img)
# print(train_ds._batch_size.numpy())

print(15*"-")
img,_ = next(iter(train_ds))
print(type(img[1]))


