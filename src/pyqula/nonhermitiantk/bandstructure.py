import numpy as np
from .. import algebra,operators

# workaround for non hermitian Hamiltonians


def get_bands_nd(h,kpath=None,operator=None,num_bands=None,
                    callback=None,central_energy=0.0,nk=400,
                    ewindow=None,eigmode="complex",
                    output_file="BANDS.OUT",write=True,
                    silent=True):
    """
    Get an n-dimensional bandstructure
    """
    if num_bands is not None:
      if num_bands>(h.intra.shape[0]-1): num_bands=None
    if operator is not None: operator = h.get_operator(operator)
    if num_bands is None: # all the bands
      if operator is not None: 
        def diagf(m): # diagonalization routine
            return algebra.eig(m) # all eigenvals and eigenfuncs
      else: 
        def diagf(m): # diagonalization routine
            return algebra.eigvals(m) # all eigenvals and eigenfuncs
    else: # using arpack
      h = h.copy()
      h.turn_sparse() # sparse Hamiltonian
      def diagf(m):
        eig,eigvec = slg.eigs(m,k=num_bands,which="LM",sigma=central_energy,
                                    tol=arpack_tol,maxiter=arpack_maxiter)
        if operator is None: return eig
        else: return (eig,eigvec)
    # open file and get generator
    hkgen = h.get_hk_gen() # generator hamiltonian
    kpath = h.geometry.get_kpath(kpath,nk=nk) # generate kpath
    def getek(k):
      """Compute this k-point"""
      out = [] # output list
      hk = hkgen(kpath[k]) # get hamiltonian
      if operator is None: # just compute the energies
        es = diagf(hk)
        es = np.sort(es) # sort energies
        for e in es:  # loop over energies
            out.append([k,e])
        if callback is not None: callback(k,es) # call the function
      else:
        es,ws = diagf(hk)
        ws = ws.transpose() # transpose eigenvectors
        def evaluate(w,k,A): # evaluate the operator
            if type(A)==operators.Operator:
                waw = A.braket(w,k=kpath[k]).real
            elif callable(A):  
              try: waw = A(w,k=kpath[k]) # call the operator
              except: 
                print("Check out the k optional argument in operator")
                waw = A(w) # call the operator
            else: waw = braket_wAw(w,A).real # calculate expectation value
            return waw # return the result
        for (e,w) in zip(es,ws):  # loop over waves
          if callable(ewindow):
              if not ewindow(e): continue # skip iteration
          if isinstance(operator, (list,)): # input is a list
              waws = [evaluate(w,k,A) for A in operator]
          else: waws = [evaluate(w,k,operator)]
          oi = [k,e] # create list
          for waw in waws:  oi.append(waw) # add this one
          out.append(oi) # store
        # callback function in each iteration
        if callback is not None: callback(k,es,ws) # call the function
      return out # return string
    if True:
      esk = [] # empty list
      for k in range(len(kpath)): # loop over kpoints
        ek = getek(k)
        esk += ek # add lists
    esk = np.array(esk).T
    if eigmode=="complex": pass # full eigenvalue
    elif eigmode=="real": esk[1] = esk[1].real # real part of eigenvalue
    elif eigmode=="imag": esk[1] = esk[1].imag # imag part of eigenvalue
    else: raise
    return esk

