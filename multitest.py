#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import time
import numpy as np
def basic_func(x):
    if x == 0:
        return 'zero'
    elif x%2 == 0:
        return 'even'
    else:
        return 'odd'

def multiprocessing_func(x):
    y = x*x
    time.sleep(2)
    print('{} squared results in a/an {} number'.format(x, basic_func(y)))
    
def lambmatrix(t,nx,ny,nz):
    from numpy import linalg as LA
    print(t)
    S = np.random.rand(3,3,nx,ny,nz)
    Ome = np.random.rand(3,3,nx,ny,nz)
    lamb2 = np.zeros((nx,ny,nz),dtype=np.float32)
    for i in range(0,nx):
                for j in range(0,ny):
                    for k in range(0,nz):
                        w = LA.eigvals((S[:,:,i,j,k],Ome[:,:,i,j,k]))
                        w = np.sort(-w)
#                         if w[1]<0:
#                             lamb2[i,j,k] = w[1]
    return lamb2    