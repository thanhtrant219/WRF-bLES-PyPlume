#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from numpy import linalg as LA
import numpy as np
def shape(dt):
    return dt.shape[0],dt.shape[1],dt.shape[2],dt.shape[3]
def transpose(x): 
    a11 = x[0,0]
    a12 = x[0,1]
    a13 = x[0,2]
    a21 = x[1,0]
    a22 = x[1,1]
    a23 = x[1,2]
    a31 = x[2,0]
    a32 = x[2,1]
    a33 = x[2,2]
    x = np.array([[a11,a21,a31],[a12,a22,a32],[a13,a23,a33]])
    return x
def muiltiply(x,y):
    a11 = x[0,0]*y[0,0]+x[0,1]*y[1,0]+x[0,2]*y[2,0]
    a12 = x[0,0]*y[0,1]+x[0,1]*y[1,1]+x[0,2]*y[2,1]
    a13 = x[0,0]*y[0,2]+x[0,1]*y[1,2]+x[0,2]*y[2,2]
    a21 = x[1,0]*y[0,0]+x[1,1]*y[1,0]+x[1,2]*y[2,0]
    a22 = x[1,0]*y[0,1]+x[1,1]*y[1,1]+x[1,2]*y[2,1]
    a23 = x[1,0]*y[0,2]+x[1,1]*y[1,2]+x[1,2]*y[2,2]
    a31 = x[2,0]*y[0,0]+x[2,1]*y[1,0]+x[2,2]*y[2,0]
    a32 = x[2,0]*y[0,1]+x[2,1]*y[1,1]+x[2,2]*y[2,1]
    a33 = x[2,0]*y[0,2]+x[2,1]*y[1,2]+x[2,2]*y[2,2]
    x = np.array([[a11,a21,a31],[a12,a22,a32],[a13,a23,a33]])
    return x
#trace of 3x3 matrix(AT A)
def trace(x):
    tra = (x[0,0]*x[1,1]*x[2,2])
    return tra
def lambmatrix(S,Ome,nx,ny,nz):
    from numpy import linalg as LA
    lamb2 = np.zeros((nx,ny,nz),dtype=np.float32)
    for i in range(0,nx):
                for j in range(0,ny):
                    for k in range(0,nz):
                        w = LA.eigvals(S[:,:,i,j,k].dot(S[:,:,i,j,k])+Ome[:,:,i,j,k].dot(Ome[:,:,i,j,k]))
                        w = np.sort(-w)
                        if w[1]<0:
                            lamb2[i,j,k] = w[1]
    return lamb2
def lambda2(datau,datav,dataw,t,dx=40,dy=40,dz=10,dt=10):
    from numpy import linalg as LA
    nx,ny,nz,nt = shape(datau)         
#     percent = np.str(np.round(t/nt*100))+"%"
#     print('{}\r'.format(percent), end="")
    print(t)
        
    gu = np.gradient(datau[:,:,:,t],dx,dy,dz)
    gv = np.gradient(datav[:,:,:,t],dx,dy,dz)
    gw = np.gradient(dataw[:,:,:,t],dx,dy,dz)             
    J = np.array([gu,gv,gw])
    JT = transpose(J)
    S = (J+JT)*0.5
    Ome = (J-JT)*0.5
    lamb2 = lambmatrix(S,Ome,nx,ny,nz)
#     print('{}\r'.format("100% Completed"), end="")
    return lamb2

