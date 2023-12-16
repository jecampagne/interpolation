import numpy as np
from functools import partial
from scipy.special import sici


# Different interpolant in Real & Fourier space

def Unity(x,ker,Jmax=5):
  js = np.arange(-Jmax,Jmax+1)
  X,Y = np.meshgrid(x,js)
  kxpj = ker(X+Y)
  return np.sum(kxpj,axis=0)

# Whittaker-Shannon sinc. interpolant
def sincInterp(x):
  return np.sinc(x)
def hatSincInterp(u):
  u = np.abs(u)
  return  np.piecewise(u,[u<0.5,u==0.5,u>0.5],[lambda x: 1., lambda x:0.5, lambda x:0])


##
#  BSpline & BSpline cardinal
##

def Bspline3(x):
  # nb. degre poly is 3
  x = np.abs(x)
  def p1(x):
    return 2/3+1/2* x * x * (x-2)
  def p2(x):
    return -1/6 *(x-2)*(x-2)*(x-2)
  return np.piecewise(x, [x<1., (x>=1)&(x<2.), x>=2],[lambda x:p1(x),lambda x:p2(x),lambda x: 0.])

def hatBspline3(u):
  s = np.sinc(u)
  s2 = s*s
  return s2 * s2

def hatBsplineCard3(u):
  sinpu = np.sin(np.pi*u)
  return hatBspline3(u)/(1-2/3*sinpu*sinpu)

def s3(n):
    n = n.astype(int)
    r3 = np.sqrt(3)
    return (-2*(n%2)+1)*r3/((2+r3)**(np.abs(n)))

def BsplineCard3(xin):
  xin = np.atleast_1d(xin)
  xshape=xin.shape
  x = xin.flatten()
  nmin = np.ceil(x-2).astype(int)
  nmax = np.floor(x+2).astype(int)
  res = []
  for i,x0 in enumerate(x):
    res.append(np.sum(np.array([s3(n)*Bspline3(x0-n) for n in np.arange(nmin[i],nmax[i]+1,1)])))
  return np.array(res).reshape(xshape)

def hatBspline5(u):
  s = np.sinc(u)
  s2 = s*s
  return s2 * s2 * s2

def hatBsplineCard5(u):
  b1 = 13/60
  b2 = 1/120
  b0 = 1 - b1 - b2
  piu = np.pi * u
  return hatBspline5(u)/(b0 + 2*b1*np.cos(2*piu) +2*b2*np.cos(4*piu))

###
# Lanczos interpolant of degree m as well as the background conservation corrected version (1st order)
###
def box(x):
  x = np.abs(x)
  return np.piecewise(x,[x<0.5,x==0.5,x>0.5],[lambda x: 1., lambda x:0.5, lambda x:0])

def siInt(x):
  si, _ = sici(x)
  return si

def lanczos(x,m):
  x = np.abs(x)
  return np.sinc(x)*np.sinc(x/m)*box(x/(2*m))

def hatlanczos(u,m):
  # the formula A12 are not correct.
  # Here Galsim code  https://github.com/GalSim-developers/GalSim/blob/releases/2.5/src/Interpolant.cpp
  # and validated with Mathematica
  u = np.abs(u)
  vp = m*(2*u+1)
  vm = m*(2*u-1)
  arg1 = vp+1
  arg2 = vp-1
  arg3 = vm-1
  arg4 = vm+1
  res = arg1*siInt(np.pi*arg1) - arg2*siInt(np.pi*arg2) + arg3*siInt(np.pi*arg3) - arg4*siInt(np.pi*arg4)
  return res/(2*np.pi)

def lanczos_norm(x,m):
  c1 = hatlanczos(1,m)
  return lanczos(x,m)*(1-2*c1*(np.cos(2*np.pi*x)-1)) # formula (24)

def hatlanczos_norm(u,m):
  c1 = hatlanczos(1,m)
  return (1+2*c1)*hatlanczos(u,m) - c1*(hatlanczos(u+1,m)+hatlanczos(u-1,m))  # formula (25)

lanczos_raw3 = partial(lanczos,m=3)
hatlanczos_raw3 = partial(hatlanczos,m=3)
lanczos_raw4 = partial(lanczos,m=4)
hatlanczos_raw4 = partial(hatlanczos,m=4)
lanczos_raw5 = partial(lanczos,m=5)
hatlanczos_raw5 = partial(hatlanczos,m=5)

lanczos3 = partial(lanczos_norm,m=3)
hatlanczos3 = partial(hatlanczos_norm,m=3)

lanczos4 = partial(lanczos_norm,m=4)
hatlanczos4 = partial(hatlanczos_norm,m=4)

lanczos5 = partial(lanczos_norm,m=5)
hatlanczos5 = partial(hatlanczos_norm,m=5)

###
# linear interpolant
###
def h1(x):
  x = np.abs(x)
  return np.piecewise(x,[x<1.,x>=1.],[lambda x: 1-x, lambda x:0])
def hat1(u):
  u = np.abs(u)
  s = np.sinc(u)
  return s*s

###
# cubic interpolant
###
def h3(x):
  # warning this is interpolator with perfect match at interger values
  x = np.abs(x)
  def p1(x): # x<1
    return 1+x*x*(1.5*x-2.5)
  def p2(x):
    return -0.5*(x-1.0)*(x-2.0)*(x-2.0)
  return np.piecewise(x, [x<1., (x>=1)&(x<2.), x>=2],[lambda x:p1(x),lambda x:p2(x),lambda x: 0.])

def hat3(u):
  # warning this is interpolator with perfect match at interger values
  u = np.abs(u)
  s = np.sinc(u)
  c = np.cos(np.pi * u)
  return s * s * s * (3.0 * s - 2.0 * c)

###
# Version by Erik H. W. Meijering, Karel J. Zuiderveld, and Max A. Viergever
# IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 8, NO. 2, FEBRUARY 1999 p 192
###
def h5MZVgen(x,c):
  x = np.abs(x)
  def p1(x): # x<1
    return 1. + x*x*(-5./2.+8*c + x*x*(45./16.-18*c + x*(-21./6.+10.*c)))
  def p2(x): # 1<=x<2
    return 5. + x*(-15. + x*(35./2. + x*(-10. + (45./16. - (5.*x)/16.)*x))) + \
      c * (-66. + x*(265. + x*(-392. + x*(270. + x*(-88. + 11.*x)))))
  def p3(x): # 2<=x<3
    return c*(-162. + x*(297. + x*(-216. + x*(78. + (-14. + x)*x))))
  return np.piecewise(x, [x<1., (x>=1)&(x<2.), (x>=2)&(x<3.), x>=3.],[lambda x:p1(x),lambda x:p2(x),lambda x:p3(x),lambda x: 0.])

def h5MZV(x):
  x = np.abs(x)
  def p1(x): # x<1
    return 1. + x*x * (-(17./8.) + (63./32. - (27.* x)/32.)*x*x)
  def p2(x): # 1<=x<2
    return 1./64.*(-2. + x) * (-1. + x) * (61. + x* (9. + x* (-45. + 13.* x)))
  def p3(x): # 2<=x<3
    return 3./64.* (-3. + x)*(-3. + x)*(-3. + x)*(-3. + x)*(-2. + x)
  return np.piecewise(x, [x<1., (x>=1)&(x<2.), (x>=2)&(x<3.), x>=3.],[lambda x:p1(x),lambda x:p2(x),lambda x:p3(x),lambda x: 0.])


def hat5MZV(u):
  u = np.abs(u)
  def fapp(u):
    u2 = u*u
    u4 = u2*u2
    P = 0.33333333333333333333 - 0.24271404762585718748 * u2 + 0.039750660445573631826 * u4
    Q = 1.0000000000000000000  + 0.25881829723136429944 * u2 + 0.026805791044710590408 * u4
    return P/Q

  def fno_app(u):
    piu = np.pi * u
    piu2 = piu*piu
    s = np.sinc(u)
    c = np.cos(piu)
    return (s-c)/piu2

  def f(u):
    return np.piecewise(u,[u<0.1,u>=0.1],[lambda u: fapp(u), lambda u: fno_app(u)])

  piu = np.pi * u
  piu2 = piu*piu
  c = np.cos(piu)
  s = np.sinc(u)
  s2 = s*s
  s3 = s*s2
  return 3./8. * s3 *( 35. * f(u) + s2 * (6.*c -15.*s) )


###
# quitic interpolants
# Gary M. Bernstein, Daniel Gruen
# https://arxiv.org/pdf/1401.2636v2.pdf
###
def h5Gary(x):
  x = np.abs(x)
  def p1(x): # x<1
    return 1.0 + x*x*x*(-95./12. + x*(23./2. + x*(-55./12.)))
  def p2(x): # 1<=x<2
    return (x-1.)*(x-2.)*(-23./4. + x*(29./2. + x*(-83./8. + x*(55./24.))))
  def p3(x): # 2<=x<3
    return (x-2.)*(x-3.)*(x-3.)*(-9./4. + x*(25./12. + x*(-11./24.)))
  return np.piecewise(x, [x<1., (x>=1)&(x<2.), (x>=2)&(x<3.), x>=3.],[lambda x:p1(x),lambda x:p2(x),lambda x:p3(x),lambda x: 0.])

def hat5Gary(u):
  u = np.abs(u)
  s = np.sinc(u)
  s2 = s*s
  s4 = s2*s2
  s5 = s*s4
  piu = np.pi * u
  c = np.cos(piu)
  piu2 = piu*piu
  return s5 * (2.0 * (-27.0 + piu2) * c + s * (55.0 -19.0 * piu2))

###
# JE : variation on Bernstein &  Gruen quintic interpolant  to get \hat{K}(u) approx (u-1)^6
###
def h5JE(x):
  x = np.abs(x)
  def p1(x): # x<1
    x2 = x * x
    pi2 = np.pi * np.pi
    return (15. *(-12. + x2*(27. + x*(-13. + (-3. + x)*x))) \
            + pi2 * (12. - x2*(15. + x*(35. + x*(-63. + 25.*x)))))/(12.*(-15. +pi2))
  def p2(x): # 1<=x<2
    pi2 = np.pi * np.pi
    return ((-2. + x) * (-1 + x)*(-15.*(24. + x*(-3. + (-6. + x)*x))\
                + pi2 * (-48. + x * (153. + x*(-114. + 25.*x)))))/(24.*(-15. + pi2))
  def p3(x): # 2<=x<3
    pi2 = np.pi * np.pi
    return -(((-3. + x)*(-3.+x)*(-2. + x)*(-3.* (-7. + x)* x\
                + pi2*(-3. + x)*(-8. + 5.*x)))/(24.*(-15. + pi2)))
  return np.piecewise(x, [x<1., (x>=1)&(x<2.), (x>=2)&(x<3.), x>=3.],\
             [lambda x:p1(x),lambda x:p2(x),lambda x:p3(x),lambda x: 0.])

def hat5JE(u):
  u = np.abs(u)
  pi2 = np.pi * np.pi
  piu = np.pi * u
  piu2 = piu*piu
  c = np.cos(piu)
  ss = np.sin(piu)
  s = np.sinc(u)
  s2 = s*s
  s4 = s2*s2
  s5 = s*s4
  return (s5* (np.pi*(24.*np.pi* (-1. + u*u)*c - (39. + 7.*pi2)* u*ss)\
               + 5.*(-3. + 5.*pi2)*s))/(-15. + pi2)


###
# JE: 7-th order (septic) piecewise ploynomial interpolant of the same familly of h5GB & h5JE
def h7JE(x):
  x = np.abs(x)
  def p1(x): # x<1
    x2 = x * x
    return 1 + (x**2*(-196 + x**2*(-959 + x*(2569 + x*(-2181 + 623*x)))))/144.
  def p2(x): # 1<=x<2
    return -((-2 + x)*(-1 + x)*(-3312 + x*(12266 + x*(-18564 + x*(13481 + x*(-4674 + 623*x))))))/240.
  def p3(x): # 2<=x<3
    return ((-3 + x)*(-2 + x)*(-52800 + x*(111694 + x*(-93340 + x*(38421 + x*(-7790 + 623*x))))))/720.
  def p4(x): # 3<=x<4
    return -((-4 + x)**4*(-3 + x)*(681 + x*(-490 + 89*x)))/720.
  return np.piecewise(x, [x<1., (x>=1)&(x<2.), (x>=2)&(x<3.), (x>=3)&(x<4.), x>=4.],\
             [lambda x:p1(x),lambda x:p2(x),lambda x:p3(x),lambda x:p4(x),lambda x: 0.])

def hat7JE(u):
  u = np.abs(u)
  pi2 = np.pi * np.pi
  piu = np.pi * u
  piu2 = piu*piu
  c = np.cos(piu)
  ss = np.sin(piu)
  s = np.sinc(u)
  s2 = s*s
  s4 = s2*s2
  s7 = s*s2*s4
  return s7*(2*(-933+58*piu2)*c + (1869-734*piu2)*s)/3.


##### 
# Magic Kernel used in Facebook & Instagram https://johncostella.com/magic/mks.pdf
# JE: this is an approx of B-Spline cardinal
def mks2021(x):
  x = np.abs(x)
  def p1(x): # x<1/2
    return 577./576. - 239./144. * x*x
  def p2(x): #1/2<=x<3/2
    return (239. - 379. * x + 140. * x*x )/144.
  def p3(x): #3/2<=x<5/2
    return (-130. + 113. * x - 24. * x*x )/144.
  def p4(x): #5/2<=x<7/2
    return (45. - 27.* x + 4. * x*x )/144.
  def p5(x): #7/2<=x<9/2
    return (-81. + 36. * x - 4. * x*x)/1152.
  return np.piecewise(x,
            [x<0.5, (x>=0.5)&(x<1.5), (x>=1.5)&(x<2.5), (x>=2.5)&(x<3.5), (x>=3.5)&(x<4.5), x>=4.5],
             [lambda x:p1(x),lambda x:p2(x),lambda x:p3(x),lambda x:p4(x),lambda x:p5(x),lambda x: 0.])

def hatmks2021(u):
  u = np.abs(u)
  piu = np.pi *u
  s = np.sinc(u)
  return s*s*s*(102. - 35.*np.cos(2*piu) + 6.* np.cos(4*piu) - np.cos(6*piu))/72.