old = False # Patches are correct

import tensorflow as tf
image = tf.constant([[[1],   [2],  [3],  [4]],
                 [[5],   [6],  [7],  [8]],
                 [[9],  [10], [11],  [12]],
                [[13], [14], [15],  [16]]])

patch_size = [1,2,2,1]
if old:
  strides = patch_size
else:
  strides = [1,1,1,1]

patches = tf.extract_image_patches([image], patch_size, strides, [1, 1, 1, 1], 'VALID')
# replace patch_size, patch_size with
# ksizes = ksizes, strides= strides

if old:
  patches = tf.reshape(patches, [4, 2, 2, 1])
else:
  patches = tf.reshape(patches, [9, 2, 2, 1])

if old:
  reconstructed = tf.reshape(patches, [1, 4, 4, 1])
  rec_new = tf.space_to_depth(reconstructed,2)
  rec_new = tf.reshape(rec_new,[4,4,1])
else:
  reconstructed = tf.reshape(patches, [1, 6, 6, 1])
  rec_new = tf.space_to_depth(reconstructed,2)
  rec_new = tf.reshape(rec_new,[6,6,1])

sess = tf.Session()
I,P,R_n = sess.run([image,patches,rec_new])
print(I)
print(I.shape)
print(P.shape)
print(R_n)
print(R_n.shape)
