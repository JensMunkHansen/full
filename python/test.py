import numpy as np
import rolling_window as lib

bum = np.reshape(np.array(10*[np.arange(25)]).T, (5,5,10))

bum0 = lib.rolling_window(bum, (2,2,0), wsteps=(1,1,0))
# Screws up
bum1 = lib.rolling_window(bum, (2,2,0), wsteps=(1,1,0),toend=False)

# steps are (1,1)

old_strides = np.array(bum.strides)
wsteps = np.r_[1, 1]
new_strides = old_strides[0:2] * wsteps

new_strides = np.concatenate((new_strides, old_strides))
new_shape = np.r_[4,4,2,2,10]

bum2 = np.lib.stride_tricks.as_strided(bum, shape=new_shape, strides=new_strides)
