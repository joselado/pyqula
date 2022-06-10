import numpy as np
from ..green import gauss_inverse # calculate the desired green functions
from ..algebra import dagger,sqrtm

delta_smatrix = 1e-12 # delta for the smatrix



def get_smatrix(ht,energy=0.0,as_matrix=False,check=True):
    """Calculate the S-matrix of an heterostructure"""
    # now do the Fisher Lee trick
    delta = ht.delta
    smatrix = [[None,None],[None,None]] # smatrix in list form
    # get the selfenergies, using the same coupling as the lead
    selfl = ht.get_selfenergy(energy,delta=delta,lead=0,pristine=True)
    selfr = ht.get_selfenergy(energy,delta=delta,lead=1,pristine=True)
    # get the central Green's function
    gmatrix = ht.get_central_gmatrix(selfl=selfl,selfr=selfr,
                                   energy=energy)
    # gamma functions
    test_gauss = True # gauss only works with square matrices
    gammar = 1j*(selfr-dagger(selfr))
    gammal = 1j*(selfl-dagger(selfl))
    # calculate the relevant terms of the Green function
    g11 = gauss_inverse(gmatrix,0,0,test=test_gauss)
    g12 = gauss_inverse(gmatrix,0,-1,test=test_gauss)
    g21 = gauss_inverse(gmatrix,-1,0,test=test_gauss)
    g22 = gauss_inverse(gmatrix,-1,-1,test=test_gauss)
    ######## now build up the s matrix with the fisher trick
    # the identity can have different dimension ignore for now....
    iden = np.matrix(np.identity(g11.shape[0],dtype=complex)) # create identity
    iden11 = np.matrix(np.identity(g11.shape[0],dtype=complex)) # create identity
    iden22 = np.matrix(np.identity(g22.shape[0],dtype=complex)) # create identity
    smatrix[0][0] = -iden + 1j*sqrtm(gammal)@g11@sqrtm(gammal) # matrix
    smatrix[0][1] = 1j*sqrtm(gammal)@g12@sqrtm(gammar) # transmission matrix
    smatrix[1][0] = 1j*sqrtm(gammar)@g21@sqrtm(gammal) # transmission matrix
    smatrix[1][1] = -iden + 1j*sqrtm(gammar)@g22@sqrtm(gammar) # matrix
    if check: # check whether the matrix is unitary
        from .unitarize import check_and_fix
        smatrix = check_and_fix(smatrix,error=100*delta)
    if as_matrix:
      from scipy.sparse import bmat,csc_matrix
      smatrix2 = [[csc_matrix(smatrix[i][j]) for j in range(2)] for i in range(2)]
      smatrix = bmat(smatrix2).todense()
    return smatrix


def get_central_gmatrix(ht,selfl=None,selfr=None,energy=0.0):
    """Return the inverse of the central Green's function"""
    delta = ht.delta
    if selfl is None: selfl = ht.get_selfenergy(energy,
        delta=delta,lead=0,pristine=True)
    if selfr is None: selfr = ht.get_selfenergy(energy,
        delta=delta,lead=1,pristine=True)
    if ht.block_diagonal:
        ht2 = enlarge_hlist(ht) # get the enlaged hlist with the leads
        gmatrix = effective_tridiagonal_hamiltonian(ht2.central_intra,
                                      selfl,selfr,
                                      energy=energy,
                                      delta=delta + ht.extra_delta_central)
    else: # not block diagonal
        gmatrix = build_effective_hlist(ht,energy=energy,delta=delta,
                                       selfl=selfl,
                                      selfr=selfr)
    return gmatrix




def effective_tridiagonal_hamiltonian(intra,selfl,selfr,
                                        energy = 0.0, delta=1e-5):
    """ Calculate effective Hamiltonian"""
    if not type(intra) is list: raise # assume is list
    n = len(intra) # number of blocks
    iout = [[None for i in range(n)] for j in range(n)] # empty list
    iden = np.matrix(np.identity(intra[0][0].shape[0],dtype=np.complex))
    if delta>delta_smatrix: delta = delta_smatrix # small delta is critical!
    ez = iden*(energy +1j*delta) # complex energy
    for i in range(n):
      iout[i][i] = ez - intra[i][i] # simply E -H
    for i in range(n-1):
      iout[i][i+1] = -intra[i][i+1] # simply E -H
      iout[i+1][i] = -intra[i+1][i] # simply E -H
    # and now the selfenergies
    iout[0][0] = iout[0][0] -selfl
    iout[-1][-1] = iout[-1][-1] -selfr
    return iout




def enlarge_hlist(ht):
    """Add a single cell of the leads to the central part"""
    ho = ht.copy() # copy heterostructure
    if not ht.block_diagonal: raise # check that is in block diagonal form
    nc = len(ht.central_intra) # number of cells in the central
    hcentral = [[None for i in range(nc+2)] for j in range(nc+2)]
    for i in range(nc): # intraterm
      hcentral[i+1][i+1] = ht.central_intra[i][i].copy() # common
    for i in range(nc-1): # interterm
      hcentral[i+1][i+2] = ht.central_intra[i][i+1].copy() # common
      hcentral[i+2][i+1] = ht.central_intra[i+1][i].copy() # common
    # now the new terms
    hcentral[0][0] = ht.left_intra.copy() # left
    hcentral[-1][-1] = ht.right_intra.copy() # right
    if nc>0: # more than two cells in the center
        hcentral[0][1] = dagger(ht.left_coupling)*ht.scale_lc # left
        hcentral[1][0] = ht.left_coupling*ht.scale_lc # left
        hcentral[-2][-1] = ht.right_coupling*ht.scale_rc # right
        hcentral[-1][-2] = dagger(ht.right_coupling)*ht.scale_rc # right
    else: # no original central part
        # here the average of the two hoppings will be performed
        # if the Hamiltonian of the scattering region is not provided
        # this may be not the optimal intuitive choice
        # perhaps square root of the product is more natural
        # gamma = sqrtm(dagger(ht.left_coupling)@ht.right_coupling)?
        hcentral[0][1] = dagger(ht.left_coupling) # left
        hcentral[1][0] = ht.left_coupling.copy() # left
        hcentral[-2][-1] += ht.right_coupling # right
        hcentral[-1][-2] += dagger(ht.right_coupling) # right
        hcentral[0][1] *= ht.scale_rc*ht.scale_lc/2. # factor 1/2 for DC
        hcentral[1][0] *= ht.scale_rc*ht.scale_lc/2. # factor 1/2 for DC
    # store in the object
    ho.central_intra = hcentral
    # and redefine the new lead couplings
    ho.right_coupling = ht.right_inter
    ho.left_coupling = ht.left_inter
    return ho # return the new heterostructure











def build_effective_hlist(ht,energy=0.0,delta=0.0001,selfl=None,selfr=None):
    """ Calculate list of effective Hamiltonian which will be inverted"""
    if (selfl is None) or (selfr is None):
        (selfl,selfr) = get_surface_selfenergies(ht,energy=energy,delta=delta,
                        pristine=True)
    intra = ht.central_intra # central intracell hamiltonian
    if len(ht.central_intra)==0: # no central part provided
        ht.central_intra = (ht.left_intra + ht.right_intra)/2.
        print("Generating a dummy central cell, you may want to use a different geometry")
    ce = energy +1j*delta
    idenc = np.matrix(np.identity(len(ht.central_intra),dtype=complex))*ce
    idenl = np.matrix(np.identity(len(ht.left_intra),dtype=complex))*ce
    idenr = np.matrix(np.identity(len(ht.right_intra),dtype=complex))*ce
    hlist = [[None for i in range(3)] for j in range(3)] # list of matrices
    # set up the different elements
    # first the intra terms
    hlist[0][0] = idenl - ht.left_intra - selfl
    hlist[1][1] = idenc - ht.central_intra
    hlist[2][2] = idenr - ht.right_intra - selfr
    # now the inter cell
    hlist[0][1] = -dagger(ht.left_coupling)*ht.scale_lc
    hlist[1][0] = -ht.left_coupling*ht.scale_lc
    hlist[2][1] = -dagger(ht.right_coupling)*ht.scale_rc
    hlist[1][2] = -ht.right_coupling*ht.scale_rc
#    for h in hlist: print(h)
    return hlist

