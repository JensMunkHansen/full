old = False # Patches are correct
#old = True

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

patches = tf.extract_image_patches([image], patch_size, strides, [1, 1, 1, 1], 'VALID') # (1,2,2,4)
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
  rec_new = tf.reshape(rec_new,[1,6,6,1])

  # Conv with size (1,1) and stride (2,2)
  k = tf.constant([[1.0]],dtype=tf.float32, name='k')
  kernel = tf.reshape(k, [1, 1, 1, 1], name='kernel')
  bum = tf.to_float(rec_new)

  # Not good  - pick upper left instead
  #rec_new = tf.squeeze(tf.nn.conv2d(bum, kernel, [1, 2, 2, 1], "SAME"))
  # Nearest is too random
  rec_new = tf.squeeze(tf.image.resize_nearest_neighbor(bum, (4,4)))


sess = tf.Session()
I,P,R_n = sess.run([image,patches,rec_new])
print(I)
print(I.shape)
print(P.shape)
print(R_n)
print(R_n.shape)


def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def pool(value, name='pool'):
    """Downsampling operation.
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, d0/2, d1/2, ..., dn/2, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        out = value
        for sh_i in sh[1:-1]:
            assert sh_i % 2 == 0
        for i in range(len(sh[1:-1])):
            out = tf.reshape(out, (-1, 2, np.prod(sh[i + 2:])))
            out = out[:, 0, :]
        out_size = [-1] + [math.ceil(s / 2) for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out
