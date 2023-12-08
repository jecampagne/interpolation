import numpy as np
from  interp1DKernels  import *
from functools import partial

# Approx of the Sinc periodic Eq 10 of 
# Gary M. Bernstein & Daniel Gruen 2014 article
# https://arxiv.org/pdf/1401.2636v2.pdf

###
# "sinc periodic kernel" aka the "ideal" kernel
# In practice N is the length of the DFT (1D)
###
def KuSincWrapped(u,N):
  return np.exp(1j * np.pi * u) * np.sinc(N*u)/np.sinc(u) # Eq. 10

###
# Approx using the Lanczos-m (non renormalized) kernel
###
def lanczos_p(x,m,N):
  x = np.abs(x)
  th=m/N
  return np.piecewise(x, [x<th,x>=th],[lambda x: lanczos(N*x,m)/lanczos(x,m), lambda x:0])

def rlanczosWrapped(u,N,m=3):
    return np.sign(0.5-np.floor(u+0.5)%2) * lanczos_p(u-np.floor(u+0.5),m,N)

def lanczosWrapped(u,N,m=3):
    return np.exp(1j * np.pi * u) * rlanczosWrapped(u,N,m)

lanczos3Wrapped = partial(lanczosWrapped,m=3)
lanczos5Wrapped = partial(lanczosWrapped,m=5)

###
# Approx using the Cubic kernel
###
def cubic_p(x,N):
  x = np.abs(x)
  th=2./N
  return np.piecewise(x, [x<th,x>=th],[lambda x: h3(N*x)/h3(x), lambda x:0])

def rcubicWrapped(u,N):
    return np.sign(0.5-np.floor(u+0.5)%2)*cubic_p(u-np.floor(u+0.5),N)

def cubicWrapped(u,N):
    return np.exp(1j * np.pi * u) * rcubicWrapped(u,N)


###
# Approx using the Quintic kernel proposed by Bernstein & Gruen
###
def quinticBG_p(x,N):
  x = np.abs(x)
  th=3./N
  return np.piecewise(x, [x<th,x>=th],[lambda x: h5Gary(N*x)/h5Gary(x), lambda x:0])

def rquinticBGWrapped(u,N):
    return np.sign(0.5-np.floor(u+0.5)%2)*quinticBG_p(u-np.floor(u+0.5),N)

def quinticBGWrapped(u,N):
    return np.exp(1j * np.pi * u) * rquinticBGWrapped(u,N)

###
# Approx using the Quintic kernel proposed by JE
###
def quinticJE_p(x,N):
  x = np.abs(x)
  th=3./N
  return np.piecewise(x, [x<th,x>=th],[lambda x: h5JE(N*x)/h5JE(x), lambda x:0])

def rquinticJEWrapped(u,N):
    return np.sign(0.5-np.floor(u+0.5)%2)*quinticJE_p(u-np.floor(u+0.5),N)

def quinticJEWrapped(u,N):
    return np.exp(1j * np.pi * u) * rquinticJEWrapped(u,N)

##
#Generic  NOT YET VALIDATED
## 
def k_p(x,N,ker,xmax):
  x = np.abs(x)
  th=xmax/N
  return np.piecewise(x, [x<th,x>=th],[lambda x: h5JE(N*x)/h5JE(x), lambda x:0])

def rkWrapped(u,N,ker,xmax):
  return np.sign(0.5-np.floor(u+0.5)%2)*k_p(u-np.floor(u+0.5),N,ker,xmax)

def kernelWrapped(u,N,ker,umax):
  return np.exp(1j * np.pi * u) * kWrapped(u,N,ker,xmax)
    