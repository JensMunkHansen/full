import numpy as np
import tensorly as tl

X = tl.tensor(np.arange(24).reshape((3, 4, 2)))



tl.unfold(X, 0) # mode-1 unfolding

tl.unfold(X, 1) # mode-2 unfolding

tl.unfold(X, 2) # mode-3 unfolding




unfolding = tl.unfold(X, 1)
original_shape = X.shape
tl.fold(unfolding, 1, original_shape)







my_matrix = np.random.rand(40, 40);

upper_half = np.hsplit(np.vsplit(my_matrix, 2)[0], 2)
lower_half = np.hsplit(np.vsplit(my_matrix, 2)[1], 2)

c11 = upper_half[0]
c12 = upper_half[1]
c21 = lower_half[0]
c22 = lower_half[1]

#Bonus to recombine them using numpy:

c = np.vstack([np.hstack([c11, c12]), np.hstack([c21, c22])])
