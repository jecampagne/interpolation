import numpy as np

# 1D & 2D FFT convention used in
# Gary M. Bernstein & Daniel Gruen 2014 article
# https://arxiv.org/pdf/1401.2636v2.pdf
# translated into numpy fft code


# The conclusion is : 
# we can use npdft2Dshift and npinvdft2Dshift to perform Gary et al 2D FFT and inverse 2D FFT.

### 1D

def dft1D(x):
  # implement a numpy FFT for pedestrian
  N = x.shape[0]
  mtx = np.array([np.exp(-2.0*1j*np.pi * m * k/N) for m in range(N) for k in range(N)]).reshape(N,-1)
  #print(mtx.shape, N)
  return mtx @ x

def invdft1D(x):
  # implement the numpy iFFT for pedestrian
  N = x.shape[0]
  mtx = np.array([np.exp(2.0*1j*np.pi * m * k/N) for m in range(N) for k in range(N)]).reshape(N,-1)
  return (mtx @ x)/N

def dft1Dshift(x):
  # implement the Gary et al FFT convention
  N = x.shape[0]
  mtx = np.array([np.exp(-2.0*1j*np.pi * m * k/N) for m in range(-N//2,N//2) for k in range(-N//2,N//2)]).reshape(N,-1)
  #print(mtx.shape, N)
  return mtx @ x

def invdft1Dshift(x):
  # implement the Gary et al iFFT
  N = x.shape[0]
  mtx = np.array([np.exp(2.0*1j*np.pi * m * k/N) for m in range(-N//2,N//2) for k in range(-N//2,N//2)]).reshape(N,-1)
  #print(mtx.shape, N)
  return (mtx @ x)/N


def npdft1Dshift(x):
  # implement Gary et al 1D DFT using numpy FFT
  return np.fft.fftshift(np.fft.fft(x))*np.array([(-1)**k for k in range(-x.shape[0]//2,x.shape[0]//2)])

def npinvdft1Dshift(x):
  # mimic Gary et al 1D i DFT using numpy iFFT
  return np.fft.ifftshift(np.fft.ifft(x))*np.array([(-1)**k for k in range(-x.shape[0]//2,x.shape[0]//2)])

### 2D 

def dft2D(x):
  # implement the numpy FFT
  N1,N2 = x.shape
  mtx1 = np.array([np.exp(-2.0*1j*np.pi * n1 * k1/N1) \
                   for n1 in range(N1) for k1 in range(N1)]).reshape(N1,-1)
  mtx2 = np.array([np.exp(-2.0*1j*np.pi * n2 * k2/N2) \
                   for n2 in range(N2) for k2 in range(N2)]).reshape(N2,-1)
  return  mtx1.T @ x @ mtx2

def dft2Dshift(x):
  # implement the Gary et al FFT
  N1,N2 = x.shape
  mtx1 = np.array([np.exp(-2.0*1j*np.pi * n1 * k1/N1) \
                   for n1 in range(-N1//2,N1//2) for k1 in range(-N1//2,N1//2)]).reshape(N1,-1)
  mtx2 = np.array([np.exp(-2.0*1j*np.pi * n2 * k2/N2) \
                   for n2 in range(-N2//2,N2//2) for k2 in range(-N2//2,N2//2)]).reshape(N2,-1)
  return  mtx1.T @ x @ mtx2

def npdft2Dshift(x):
  # implement Gary et al 2D DFT using numpy FFT
  N1,N2 = x.shape
  mtxsig = np.array([(-1)**(k1+k2) for k1 in range(-N1//2,N1//2) for k2 in range(-N2//2,N2//2)]).reshape(N1,N2)
  return np.fft.fftshift(np.fft.fft2(x)) * mtxsig

def invdft2D(x):
  # implement the numpy iFFT
  N1,N2 = x.shape
  mtx1 = np.array([np.exp(2.0*1j*np.pi * n1 * k1/N1) \
                   for n1 in range(N1) for k1 in range(N1)]).reshape(N1,-1)
  mtx2 = np.array([np.exp(2.0*1j*np.pi * n2 * k2/N2) \
                   for n2 in range(N2) for k2 in range(N2)]).reshape(N2,-1)
  return  (mtx1.T @ x @ mtx2)/(N1*N2)

def invdft2Dshift(x):
  # implement the Gary et al iFFT
  N1,N2 = x.shape
  mtx1 = np.array([np.exp(2.0*1j*np.pi * n1 * k1/N1) \
                   for n1 in range(-N1//2,N1//2) for k1 in range(-N1//2,N1//2)]).reshape(N1,-1)
  mtx2 = np.array([np.exp(2.0*1j*np.pi * n2 * k2/N2) \
                   for n2 in range(-N2//2,N2//2) for k2 in range(-N2//2,N2//2)]).reshape(N2,-1)
  return  (mtx1.T @ x @ mtx2)/(N1*N2)

def npinvdft2Dshift(x):
  # implement Gary et al 2D DFT using numpy FFT
  N1,N2 = x.shape
  mtxsig = np.array([(-1)**(k1+k2) for k1 in range(-N1//2,N1//2) for k2 in range(-N2//2,N2//2)]).reshape(N1,N2)
  return np.fft.ifftshift(np.fft.ifft2(x)) * mtxsig

