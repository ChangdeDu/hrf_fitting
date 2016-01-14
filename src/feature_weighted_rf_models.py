import numpy as np
from theano import tensor as tnsr
from theano import function, scan

X = tnsr.tensor3('X')    ##model-space tensor: G x T x D
NU = tnsr.tensor3('NU')  ##feature weight tensor: G x D x V
Z = tnsr.batched_tensordot(X,NU,axes=[[2],[1]]) ##prediction tensor: G x T x V
bigmult = function([X,NU], Z)

y = tnsr.matrix('y')  ##voxel data tensor: T x V

diff = y-Z  ##difference tensor: (T x V) - (G x T x V) = (G x T x V)
sq_diff = (diff*diff).sum(axis=1)

SQD_sum = sq_diff.sum()  ##<<this is critical
grad_SQD_wrt_NU = tnsr.grad(SQD_sum,NU) ##<<the summing trick above makes this easy. 
compute_grad = function(inputs = [y,X,NU], outputs=grad_SQD_wrt_NU)

