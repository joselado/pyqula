import numpy as np

def discard_outliers(x,y):
    """Given x and y, discard outliers in y"""
    yn = y/np.median(y) # normalize to the median
    yp = np.percentile(y,90)
    xo = x[y<yp] # less than five times the median
    yo = y[y<yp] # less than five times the median
    return xo,yo


