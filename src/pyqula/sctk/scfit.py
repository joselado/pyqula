import numpy as np
from .. import klist
from . import pairing

# function to extract different SC order parameters from a Hamiltonian

#
#def fitsc(H,nk):
#    """Fit a Hamiltonian to a collection of uniform SC orders"""
#    hkgen = H.get_hk_gen() # get generator
#    ks = klist.kmesh(nk=nk) # generate klist
#    hks = [] # empty list
#    for k in ks: hks.append(hkgen(k)) # store this matrix
#    # now generate the matrices for all the SC order parameters
#    H0 = H*0. # initialize
#    SCHs = [H0.copy().add_pairing(mode=mode)]
#
