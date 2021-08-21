import numpy as np
from scipy.interpolate import interp1d

def positive_interpolator(x,y,fill=None):
    """Given x and y, return a function that interpolates
    data, assuming a positive interpolation"""
    if fill is None: fill = (y[0],y[len(y)-1]) # filled value
    f = interp1d(x,y,kind="linear",bounds_error=False,
                    fill_value=fill) # interpolation
    return lambda x: f(x)*(1.+np.sign(f(x)))/2. # only positive
