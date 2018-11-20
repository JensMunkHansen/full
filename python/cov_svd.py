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

u,s,v = np.linalg.svd(XT, full_matrices=False)
XTinv = np.linalg.inv(XT)
diff = XTinv - np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)

print(diff.max())

norm = np.dot(np.ones((1,10)),np.dot(XTinv, np.ones((10,1))))

# MatrixXd centered = mat.rowwise() - mat.colwise().mean();
# MatrixXd cov = (centered.adjoint() * centered) / double(mat.rows() - 1);
# Map<Matrix<double,Dynamic,21,RowMajor> > mat(&(all_data[0][0], all_data.size(), 21);

print(norm)
