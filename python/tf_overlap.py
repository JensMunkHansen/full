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

def overlap_simple(tensor, stride=2):
  # Reshape the tensor to allow it to be passed in to conv2d.
  reshaped = tf.reshape(tensor, [1,1,-1,1])

  # Construct the block_size filters.
  filter_dim = [1, -1, 1, 1]
  x_filt = tf.reshape(tf.constant([1., 0., 0.]), filter_dim)
  y_filt = tf.reshape(tf.constant([0., 1., 0.]), filter_dim)
  z_filt = tf.reshape(tf.constant([0., 0., 1.]), filter_dim)

  # Stride along the tensor with the above filters.
  stride_window = [1, 1, stride, 1]
  x = tf.nn.conv2d(reshaped, x_filt, stride_window, padding='VALID')
  y = tf.nn.conv2d(reshaped, y_filt, stride_window, padding='VALID')
  z = tf.nn.conv2d(reshaped, z_filt, stride_window, padding='VALID')

  # Pack the three tensors along 4th dimension.
  result = tf.stack([x, y, z], axis=4)
  # Squeeze to get rid of the extra dimensions.
  result = tf.squeeze(result)
  return result


def overlap2(tensor, block_size=(3,3), stride=(2,2)):
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




class LocalConnectedConv(Layer):
    def __init__(
        self,
        layer = None,
        filters=0,
        size=3,
        multiplexH=1,
        multiplexW=1,
        stride=1,
        overlap=True,
        act = tf.identity,
        name ='lcconv2d',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        channels = int(self.inputs.get_shape()[-1])
        inputH=int(self.inputs.get_shape()[1])
        inputW = int(self.inputs.get_shape()[2])
        if inputH%multiplexH==0 and inputW%multiplexW==0:
            print('ok')
        else:
            return 0
        CellH= int(np.floor(inputH/multiplexH))
        CellW=int(np.floor(inputW/multiplexW))
        CellHm= int(np.floor(inputH/multiplexH))
        CellWm=int(np.floor(inputW/multiplexW))
        if overlap:
            CellH = np.floor(inputH / multiplexH*2)
            CellW = np.floor(inputW / multiplexW*2)
            CellH=int(CellH)
            CellW=int(CellW)
        Wd=False
        Hd=False
        if CellH%2==0:
            Hd=True
        if CellW%2==0:
            Wd=True
        with tf.variable_scope(name) as vs:
            Welist=[]
            Bilist=[]
            for i in range(multiplexH):
                for j in range(multiplexW):

                    We = tf.get_variable(name='weights%d-%d'%(i,j), shape=[size, size, channels, filters],
                                               initializer=tf.truncated_normal_initializer(stddev=0.03),
                                               dtype=tf.float32, trainable=True)
                    bi = tf.get_variable(name='biases%d-%d'%(i,j), shape=[filters, ],
                                              initializer=tf.constant_initializer(value=0.1),
                                              dtype=tf.float32, trainable=True)
                    Welist.append(We)
                    Bilist.append(bi)
        Convij=[]
        for i in range(multiplexH):
            for j in range(multiplexW):
                ci=np.floor((i+0.5)*CellHm-0.01)+1
                cj=np.floor((j+0.5)*CellWm-0.01)+1
                if not overlap:
                    if i==0:
                        hcs=0
                        hce=hcs+CellH+size-1
                    elif i==multiplexH-1:
                        hce=inputH
                        hcs = hce-CellH-size+1
                    elif Hd:
                        hcs=ci-(CellH+size-1)/2
                        hce=ci+(CellH+size-1)/2
                    else:
                        hcs=ci-np.floor((CellH+size-1)*0.5-0.01)-1
                        hce=ci+np.floor((CellH+size-1)*0.5-0.01)
                    if j==0:
                        wcs=0
                        wce=wcs+CellW+size-1
                    elif j==multiplexW-1:
                        wce=inputW
                        wcs = wce-CellW-size+1
                    elif Wd:
                        wcs=cj-(CellW+size-1)/2
                        wce=cj+(CellW+size-1)/2
                    else:
                        wcs=cj-np.floor((CellW+size-1)*0.5-0.01)-1
                        wce=cj+np.floor((CellW+size-1)*0.5-0.01)
                else:
                    if i == 0:
                        hcs = 0
                        hce = hcs + CellH
                    elif i == multiplexH - 1:
                        hce = inputH
                        hcs = hce - CellH
                    elif Hd:
                        hcs = ci - (CellH) / 2
                        hce = ci + (CellH) / 2
                    else:
                        hcs = ci - np.floor((CellH ) * 0.5 - 0.01)-1
                        hce = ci + np.floor((CellH ) * 0.5 - 0.01)
                    if j == 0:
                        wcs = 0
                        wce = wcs + CellW
                    elif j == multiplexW - 1:
                        wce = inputW
                        wcs = wce - CellW
                    elif Wd:
                        wcs = cj - (CellW ) / 2
                        wce = cj + (CellW ) / 2
                    else:
                        wcs = cj - np.floor((CellW ) * 0.5 - 0.01)-1
                        wce = cj + np.floor((CellW) * 0.5 - 0.01)
                hcs=int(hcs)
                wcs=int(wcs)
                hce=int(hce)
                wce=int(wce)
                it=self.inputs[:,hcs:hce,wcs:wce,:]
                convtemp=tf.nn.conv2d(it,Welist[multiplexW*i+j], strides=[1, stride, stride, 1], padding='VALID')
                convtemp=tf.add(convtemp, Bilist[multiplexW*i+j])
                Convij.append(convtemp)
        convli=[]
        for i in range(multiplexH):
            convlii=[]
            for j in range(multiplexW):
                convlii.append(Convij[multiplexW * i + j])
            convli.append(convlii)
        convt=[]
        for i in range(multiplexH):
            convt.append(tf.concat(convli[i],axis=2))
        convfin=tf.concat(convt,axis=1)
        self.outputs =act(convfin)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(Welist)
        self.all_params.extend(Bilist)
