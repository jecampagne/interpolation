import numpy as np
from BG_DFT_impl import *

# test 1D
x=np.random.normal(size=(10,))
assert np.allclose(dft1D(x),np.fft.fft(x))
assert np.allclose(invdft1D(x),np.fft.ifft(x))
assert np.allclose(dft1Dshift(x),npdft1Dshift(x))
assert np.allclose(npinvdft1Dshift(x),invdft1Dshift(x))

assert np.allclose(invdft1D(dft1D(x)),x) # self consistance
assert np.allclose(dft1D(invdft1D(x)),x) # self consistance
assert np.allclose(invdft1Dshift(dft1Dshift(x)),x) # self consistance
assert np.allclose(dft1Dshift(invdft1Dshift(x)),x) # self consistance
assert np.allclose(npinvdft1Dshift(npdft1Dshift(x)),x) # self consistance
assert np.allclose(npdft1Dshift(npinvdft1Dshift(x)),x) # self consistance

# test 2D
x=np.random.normal(size=(4,6))

assert np.allclose(dft2D(x),np.fft.fft2(x))
assert np.allclose(dft2Dshift(x),npdft2Dshift(x))
assert np.allclose(invdft2D(x),np.fft.ifft2(x))
assert np.allclose(invdft2Dshift(x),npinvdft2Dshift(x))

assert np.allclose(invdft2Dshift(dft2Dshift(x)),x) # self consistance
assert np.allclose(dft2Dshift(invdft2Dshift(x)),x) # self consistance
assert np.allclose(npinvdft2Dshift(npdft2Dshift(x)),x) # self consistance
assert np.allclose(npdft2Dshift(npinvdft2Dshift(x)),x) # self consistance