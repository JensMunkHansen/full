import numpy as np

def mycov(x):
  x = x - x.mean(axis=0)
  u,s,v = np.linalg.svd(x, full_matrices=False)
  C = np.dot(np.dot(v.T,np.diag(s**2)),v)
  # First component is
  # np.dot(x, v)[:,0]
  return C / (x.shape[0] - 1)


XT = np.random.rand(10,10)

cref = np.cov(XT, rowvar=False)
c = mycov(XT)
