# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")



from pyqula import geometry
import numpy as np
g = geometry.honeycomb_zigzag_ribbon(20) # create geometry of a zigzag ribbon

def interpolate_hopping(r):
    """Modulated hopping in the ribbon as a ramp"""
    y = r[1] # take the y coordinate (in the width)
    tau = 0.02 # this is the gradient wanted
    dt = y*tau # correction to the hopping given the gradient
    # we will make a ramp of the hopping of the form
    #                   _____________
    #                  /
    #                 /
    #                /
    #               /
    #  ____________/
    # if the correction is small, change the hopping
    dtmax = 0.5 # maximum correction allowed
    if abs(dt)<dtmax: return 1.+dt # return the hopping
    elif dt>=dtmax: return 1.+dtmax # if too large, return the upper plateau
    elif dt<=-dtmax: return 1.-dtmax # if too small, return the lower plateau

def tij(r1,r2): # function for strined hoppings
    dr = r1-r2 ; dr2 = dr.dot(dr)
    if abs(dr2-1.)<1e-3: # first neighbor
        r0 = (r1+r2)/2. # average position
        return interpolate_hopping(r0) # return the hopping you want
    return 0. # otherwise

g.write_profile(interpolate_hopping,nrep=40) # write the interpolated hopping
h = g.get_hamiltonian(tij=tij) # create hamiltonian of the system
h.get_bands()







