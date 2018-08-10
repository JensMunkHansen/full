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
               'discriminator' : True,
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
plt.imshow(log_compression(env),extent=1000*np.r_[-0.02, 0.02, depths[0], depths[-1]], aspect='auto')

cfmFileTokens = glob.glob(os.path.join(datadir,'rf*.mat'))
cfmFileTokens = np.sort(cfmFileTokens)

# Sort correctly
indices = np.hstack([[0,11],np.r_[13:20],np.r_[1:10],12])

cfmFileTokens = [ cfmFileTokens[idx] for idx in indices]

rfdata = loadmat(cfmFileTokens[0])['data']

nLines = len(cfmFileTokens)
nSamples          = rfdata.shape[0]
nShotsPerEstimate = rfdata.shape[1]

rfdatas = np.zeros((nLines, nSamples, nShotsPerEstimate))

# line, sample, shot
for iLine in range(nLines):
  rfdata = loadmat(cfmFileTokens[iLine])['data']
  rfdata = rfdata / (2.0**15)
  rfdatas[iLine,:,:] = rfdata

vels = np.zeros((nLines, nSamples))

for iLine in range(nLines):
  rfdata = rfdatas[iLine]

  discriminator = np.abs(hilbert(rfdata,axis=0))

  if opt.echo:
    # Echo cancellation (average in slow-time), TODO(JMH): Improve using FIR or IIR
    rfdata = rfdata - rfdata.mean(axis=1)[:,np.newaxis]

  # After echo-cancellation
  iqdata = hilbert(rfdata,axis=0)

  if opt.discriminator:
    # Power ratio (after / before) echo-cancellation
    discriminator = 20*np.log10(np.abs(iqdata) / discriminator);

    # Averaging over pulses
    discriminator = discriminator.mean(axis=1);

  # 1D Auto-correlation
  vel = np.sum(iqdata[:,1:]*np.conj(iqdata[:,:-1]),axis=1)

  # Velocity estimate
  vel = opt.c/(2*np.pi*opt.f0) * opt.fprf/2.0 * np.angle(vel)

  if opt.discriminator:
    vel = (discriminator > opt.threshold_ratio) * vel

  vels[iLine,:] = vel

# Averaging
if opt.meanfilter:
  vels = convolve2d(vels,np.ones((3,15))/45.0,'same')

vels = vels.T

# Compute depths
depths = opt.depth_offset + opt.c/(2*opt.fs)*np.arange(iqdata.shape[0])

plt.figure()
plt.imshow(vels,extent=1000*np.r_[-0.02, 0.02, depths[0], depths[-1]],aspect='auto')

#np.sum(cv[1:]*np.conj())

# Local variables: #
# tab-width: 2 #
# python-indent: 2 #
# indent-tabs-mode: nil #
# End: #
