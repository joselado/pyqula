import numpy as np
from .. import algebra
from .. import parallel
import scipy.sparse.linalg as slg

arpack_tol = 1e-5
arpack_maxiter = 10000

def fermi_surface(h,write=True,output_file="FERMI_MAP.OUT",
                    e=0.0,nk=50,nsuper=1,reciprocal=True,
                    k0 = np.array([0.,0.]),
                    delta=None,refine_delta=1.0,operator=None,
                    mode='eigen',num_waves=10,info=False):
    """Calculates the Fermi surface of a 2d system"""
    operator = h.get_operator(operator) # get the operator
    if operator is not None: # operator given
        if not operator.linear:
            if mode=="full": mode = "eigen"
    else: # no operator given
        if mode=="full":
            operator = np.matrix(np.identity(h.intra.shape[0]))
    if h.dimensionality!=2: raise  # continue if two dimensional
    hk_gen = h.get_hk_gen() # gets the function to generate h(k)
    kxs = np.linspace(-nsuper,nsuper,nk)  # generate kx
    kys = np.linspace(-nsuper,nsuper,nk)  # generate ky
    iden = np.identity(h.intra.shape[0],dtype=np.complex128)
    kxout = []
    kyout = []
    if reciprocal: R = h.geometry.get_k2K_generator() # get function
    else:  R = lambda x: x
    # setup a reasonable value for delta
    if delta is None:  delta = 3./refine_delta*2./nk
    #### function to calculate the weight ###
    if mode=='full': # use full inversion
      def get_weight(hk,k=None):
        gf = algebra.inv((e+1j*delta)*iden - hk) # get green function
        gf = gf - algebra.inv((e-1j*delta)*iden - hk) # get green function
        if callable(operator): # callable operator
           tdos = -(operator(gf,k=k)).imag # get imaginary part
        else: tdos = -(operator*gf).imag # get imaginary part
        return np.trace(tdos).real # return trace
    elif mode=='eigen': # use full diagonalization
      def get_weight(hk,k=None):
        if operator is None:
            es = algebra.eigvalsh(hk)
            return np.sum(delta/((e-es)**2+delta**2)) # return weight
        else: # using an operator
            tmp,ds = h.get_dos(ks=[k],operator=operator,
                        write=False,
                        energies=[e],delta=delta)
            return ds[0] # return weight
    elif mode=='lowest': # use sparse diagonalization
      def get_weight(hk,k=None,**kwargs):
        if operator is None: # without operator
            es,waves = slg.eigsh(hk,k=num_waves,sigma=e,
                    tol=arpack_tol,which="LM",
                              maxiter = arpack_maxiter)
            return np.sum(delta/((e-es)**2+delta**2)) # return weight
        else: # using an operator
            htmp = h.copy() # make a dummy copy
            htmp.shift_fermi(-e) # shift the Fermi energy
            tmp,ds = htmp.get_dos(ks=[k],operator=operator,
                        write=False,
                        energies=[0.],delta=delta,
                        num_bands=num_waves)
            return ds[0] # return weight
    elif mode=='det': # use determinant method, this is not too stable
        if operator is not None: raise # not implemented
        else: # None operator
            iden = algebra.identity(h.intra)
            def get_weight(hk,k=None,**kwargs):
                hk0 = hk - e*iden # shift by the energy
                return 1./(np.abs(algebra.det(hk0))+delta)

    else: raise # unrecognized mode
  
  ##############################################
  
  
    # setup the operator
    rs = [] # empty list
    for x in kxs:
      for y in kxs:
        rs.append([x,y,0.]) # store
    k0 = np.array([k0[0],k0[1],0.])
    def getf(r): # function to compute FS
        k = R(r) + k0
        hk = hk_gen(k) # get hamiltonian
        return get_weight(hk,k=k)
    rs = np.array(rs) # transform into array
    kxout = rs[:,0] # x coordinate
    kyout = rs[:,1] # y coordinate
    if parallel.cores==1: # serial execution
        kdos = np.zeros(len(rs),dtype=np.float64) # empty array
        for ir in range(len(rs)): # loop
            kdos[ir] = getf(rs[ir]) # store
    else: # parallel execution
        kdos = parallel.pcall(getf,rs) # compute all
    if write:  # optionally, write in file
      f = open(output_file,"w") 
      for (x,y,d) in zip(kxout,kyout,kdos):
        f.write(str(x)+ "   "+str(y)+"   "+str(d)+"\n")
      f.close() # close the file
    return (kxout,kyout,np.array(kdos)) # return result


