import tensorflow as tf

def overlap(tensor, block_size=3, stride=2):
  reshaped = tf.reshape(tensor, [1,1,-1,1])

  # Construct diagonal identity matrix for conv2d filters.
  ones = tf.ones(block_size, dtype=tf.float32)
  ident = tf.diag(ones)
  filter_dim = [1, block_size, block_size, 1]
  filter_matrix = tf.reshape(ident, filter_dim)

  stride_window = [1, 1, stride, 1]

  # Save the output tensors of the convolutions
  filtered_conv = []
  for f in tf.unstack(filter_matrix, axis=1):
    reshaped_filter = tf.reshape(f, [1, block_size, 1, 1])
    c = tf.nn.conv2d(reshaped, reshaped_filter, stride_window, padding='VALID')
    filtered_conv.append(c)

  # Put the convolutions into a tensor and squeeze to get rid of extra dimensions.
  t = tf.stack(filtered_conv, axis=3)
  return tf.squeeze(t)


# Calculate the overlapping strided slice for the input tensor.
tensor = tf.constant([1, 2, 3, 4, 5, 6, 7], dtype=tf.float32)
overlap_tensor = overlap(tensor, block_size=3, stride=2)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  in_t, overlap_t = sess.run([tensor, overlap_tensor])
  print( 'input tensor:')
  print( in_t)
  print( 'overlapping strided slice:')
  print( overlap_t)
#tf.extract_image_patches
