import os
import glob
import numpy as np

from scipy.io import loadmat
from scipy.signal import (hilbert, convolve2d)

import matplotlib.pyplot as plt

from dicts import dotdict

def log_compression(img,dBrange=60):
  logenv = 20*np.log10(img/img.max())
  logenv[logenv < -dBrange] = -dBrange
  return logenv

plt.ion()

opt = dotdict({'c' : 1540.0,
               'fs' : 25e6,
               'f0' : 5e6,
               'fprf' : 10e3,
               'echo' : True,
               'depth_offset' : 0.03,
               'meanfilter' : True,
               'threshold_ratio' : -20})

filedir = os.path.dirname(os.path.realpath(__file__))

datadir = os.path.join(filedir, '../data/')

bmodeFile = glob.glob(os.path.join(datadir,'b*.mat'))[0]

rfdata = loadmat(bmodeFile)['bdata']

rfdata = rfdata / (2.0**15)

env = np.abs(hilbert(rfdata,axis=0))

# Compute depths
depths = opt.depth_offset + opt.c/(2*opt.fs)*np.arange(rfdata.shape[0])

plt.figure()
plt.imshow(log_compression(env),extent=1000*np.r_[-0.02, 0.02, depths[-1], depths[0]], aspect='auto')

cfmFileTokens = glob.glob(os.path.join(datadir,'rf*.mat'))
cfmFileTokens = np.sort(cfmFileTokens)

# Sort correctly
indices = np.hstack([[0,11],np.r_[13:20],np.r_[1:11],12])

cfmFileTokens = [ cfmFileTokens[idx] for idx in indices]

rfdata = loadmat(cfmFileTokens[0])['data']

nLines            = len(cfmFileTokens)
nSamples          = rfdata.shape[0]
nShotsPerEstimate = rfdata.shape[1]

rfdatas = np.zeros((nLines, nSamples, nShotsPerEstimate))

# line, sample, shot
for iLine in range(nLines):
  rfdata = loadmat(cfmFileTokens[iLine])['data']
  rfdata = rfdata / (2.0**15)
  rfdatas[iLine,:,:] = rfdata

vels = np.zeros((nLines, nSamples))

ueigen = np.r_[1:10]

from numpy import linalg as la

for iLine in range(nLines):
  rfdata = rfdatas[iLine] # [Fast-time, Slow-time]

  u, s, vt = la.svd(rfdata, full_matrices=False)
  xfilt = np.dot(u[:,ueigen].dot(np.diag(s[ueigen])),vt[ueigen,:])
  rfdatas[iLine,:,:] = xfilt

wsize = np.r_[3, 210]
wstep = np.r_[1, 140]
# Serious overlap
wsize = np.r_[3, 100]
wstep = np.r_[1, 50]

shape = np.array(rfdatas.shape)
old_strides = np.array(rfdatas.strides)
new_strides = old_strides[0:2] * wstep
new_strides = np.concatenate((new_strides, old_strides))

nwindows = (shape[0:2] - wsize) / wstep
new_shape = np.concatenate((nwindows,wsize, [nShotsPerEstimate]))

patches = np.lib.stride_tricks.as_strided(rfdatas, shape=new_shape, strides=new_strides)

(nx,ny) = patches.shape[0:2]

# Adjust overlap
wmask = 0.5*np.ones(patches.shape[2:4])
wmask[1:-1,:]  = 1.0

vels = np.zeros((nLines, nSamples))

#veigen = np.r_[9]
#veigen = np.r_[1:9]
veigen = np.r_[1:10]
for ix in range(nx):
  for iy in range(ny):
    patch = patches[ix,iy,:,:,:].copy()
    patch = patch.reshape((-1, nShotsPerEstimate))
    u, s, vt = la.svd(patch, full_matrices=False)
    xfilt = np.dot(u[:,veigen].dot(np.diag(s[veigen])),vt[veigen,:])
    vel = np.sqrt(np.sum(np.abs(xfilt)**2,axis=1))
    xslice = slice(ix*wstep[0], ix*wstep[0] + wsize[0], 1)
    yslice = slice(iy*wstep[1], iy*wstep[1] + wsize[1], 1)
    vels[xslice,yslice] = vels[xslice, yslice] + vel.reshape(tuple(wsize)) * wmask


# Averaging
if opt.meanfilter:
  vels = convolve2d(vels,np.ones((3,15))/45.0,'same')

vels = vels.T

# Compute depths
depths = opt.depth_offset + opt.c/(2*opt.fs)*np.arange(rfdata.shape[0])

plt.figure()
plt.imshow(vels,extent=1000*np.r_[-0.02, 0.02, depths[-1], depths[0]],aspect='auto')


#np.sum(cv[1:]*np.conj())

# Local variables: #
# tab-width: 2 #
# python-indent: 2 #
# indent-tabs-mode: nil #
# End: #
