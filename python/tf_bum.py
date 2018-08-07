import tensorflow as tf
import numpy as np

c = 3
height = 3900
width = 6000
ksizes = [1, 150, 150, 1]
strides = [1, 150, 150, 1]

image = #image of shape [1, height, width, 3]

patches = tf.extract_image_patches(image, ksizes = ksizes, strides= strides, [1, 1, 1, 1], 'VALID')
patches = tf.reshape(patches, [-1, 150, 150, 3])

reconstructed = tf.reshape(patches, [1, height, width, 3])
rec_new = tf.space_to_depth(reconstructed,75)
rec_new = tf.reshape(rec_new,[height,width,3])
