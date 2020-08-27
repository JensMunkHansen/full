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
plt.imshow(log_compression(env),extent=1000*np.r_[-0.02, 0.02, depths[0], depths[-1]], aspect='auto')

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
#ueigen = np.r_[0]
from numpy import linalg as la

for iLine in range(nLines):
  rfdata = rfdatas[iLine] # [Fast-time, Slow-time]

  u, s, vt = la.svd(rfdata, full_matrices=False)
  xfilt = np.dot(u[:,ueigen].dot(np.diag(s[ueigen])),vt[ueigen,:])
  # Perhaps, we need to mult with np.conj
  #vels[iLine,:] = np.sum(np.abs(xfilt*np.conj(xfilt)), axis=1)
  vels[iLine,:] = np.sqrt(np.sum(np.abs(xfilt)**2,axis=1))

# Averaging
if opt.meanfilter:
  vels = convolve2d(vels,np.ones((3,15))/45.0,'same')

vels = vels.T

# Compute depths
depths = opt.depth_offset + opt.c/(2*opt.fs)*np.arange(rfdata.shape[0])

plt.figure()
plt.imshow(vels,extent=1000*np.r_[-0.02, 0.02, depths[0], depths[-1]],aspect='auto')

#np.sum(cv[1:]*np.conj())

# Local variables: #
# tab-width: 2 #
# python-indent: 2 #
# indent-tabs-mode: nil #
# End: #
