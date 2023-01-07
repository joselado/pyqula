import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from ..transporttk.fermidirac import fermidirac as FD


def temperature_convolution(x,y,temp=1e-1):
    if temp==0.: return x,y
    dt = 20 # hwo many times the temperature
    x = np.array(x)
    y = np.array(y)
    fy = interp1d(x,y,bounds_error=False,fill_value=(y[0],y[len(y)-1]))
    de = temp/10
    def g(e0):
        def f(e):
            out = fy(e+e0) # call the function
            out *= FD(e-de,temp=temp) - FD(e+de,temp=temp)
            return out/de
        return quad(f,-dt*temp,dt*temp,epsrel=1e-2,limit=30)[0]/2.
    return x,np.array([g(ix) for ix in x]) # return

