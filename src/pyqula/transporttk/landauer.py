from .. import algebra
import numpy as np
from .. import green
from ..hamiltonians import is_number

dagger = algebra.dagger

# Landauer Buttiker formula

def landauer(HT,energy=0.0,error=1e-9,**kwargs):
    """ Calculates transmission using Landauer formula"""
    delta = HT.delta
    if not HT.block_diagonal:
      intra = HT.central_intra # central intraterm   
      dimhc = intra.shape[0] # dimension of the central part
    if HT.block_diagonal:
        if len(HT.central_intra)==0: # no central
            print("No central region provided")
            raise
        intra = HT.central_intra[0][0] # when it is diagonal
 # dimension of the central part
        dimhc = len(HT.central_intra)*intra.shape[0]
    iden = np.matrix(np.identity(len(intra),dtype=complex)) # create identity
    selfl = HT.get_selfenergy(energy,lead=0,delta=delta,pristine=False) # left Sigma
    selfr = HT.get_selfenergy(energy,lead=1,delta=delta,pristine=False) # right Sigma
    #################################
    # calculate Gammas 
    #################################
    gammar = 1j*(selfr-dagger(selfr))
    gammal = 1j*(selfl-dagger(selfl))
 
    #################################
    # dyson equation for the center
    #################################
    # central green function
    intra = HT.central_intra
    # full matrix
    if not HT.block_diagonal:
        heff = intra + selfl + selfr
        HT.heff = heff
        gc = (energy+1j*delta)*iden - heff
        gc = algebra.inv(gc) # calculate inverse
        G = np.trace(gammar@gc@gammal@dagger(gc)).real
        return G
    # reduced matrix
    if HT.block_diagonal:
        from copy import deepcopy
        heff = deepcopy(intra)
        heff[0][0] = intra[0][0] + selfl
        heff[-1][-1] = intra[-1][-1] + selfr
        dd = (energy+1j*delta)*iden
        for i in range(len(intra)):  # add the diagonal energy part
          heff[i][i] = heff[i][i] - dd  # this has the wrong sign!!
       # now change the sign
        for i in range(len(intra)):
          for j in range(len(intra)):
            if heff[i][j] is not None:
              heff[i][j] = -heff[i][j]
        # calculate green function
        gauss_inverse = green.gauss_inverse  # routine to invert the matrix
        # calculate only some elements of the central green function
        gcn1 = gauss_inverse(heff,len(heff)-1,0) # calculate element 1,n
        # and apply Landauer formula
        G = np.trace(gammar@gcn1@gammal@dagger(gcn1)).real
    return G # return transmission

