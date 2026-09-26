import numpy as np
import scipy.sparse.linalg as slg
from .. import algebra,operators
from ..algebra import braket_wAw

# workaround for non hermitian Hamiltonians

# same values as in the Hermitian bandstructure.py, whose arpack branch
# this one is a copy of
arpack_tol = 1e-8
arpack_maxiter = 10000


def get_bands_nd(h,kpath=None,operator=None,num_bands=None,
                    callback=None,central_energy=0.0,nk=400,
                    ewindow=None,eigmode="complex",biorthogonal=False,
                    output_file="BANDS.OUT",write=True,
                    silent=True):
    """
    Get an n-dimensional bandstructure

    operator: None, a single operator spec, or a list of operator specs
        (each spec may be a string name, matrix, Operator instance, or
        callable, exactly as accepted for a single operator). If a list is
        given, the expectation value of every operator is computed for each
        eigenstate, and the returned array gains one extra row per operator
        (k, e, c1, c2, ...) instead of just (k, e, c).

    write/output_file: as in the Hermitian get_bands_nd. The eigenvalues
        are complex here, so with eigmode="complex" the file carries the
        real and the imaginary part in two columns (k, Re e, Im e, ...)
        rather than dropping half of the eigenvalue; with eigmode="real"
        or "imag" it carries the single part that was asked for.

    biorthogonal: how an operator weighs each eigenstate. False (the
        default) takes the right eigenvector alone, <R_n|A|R_n>/<R_n|R_n>,
        which is real for a Hermitian A. True takes the left eigenvector
        too, <L_n|A|R_n>/<L_n|R_n>, which is complex in general, and whose
        sum over the states is Tr A exactly; it is the weight that the
        Green's function (w - H)^-1 = sum_n |R_n><L_n|/(w - E_n) gives
        each pole. The left eigenvectors are taken as the rows of R^-1, R
        the matrix of right eigenvectors, which keeps them biorthonormal to
        the right ones inside a degenerate level as well; close to an
        exceptional point R is ill conditioned and these weights grow
        large and cancel between the states that coalesce.
    """
    if biorthogonal and num_bands is not None:
        raise NotImplementedError("the biorthogonal weights need every "
                "left eigenvector, which the ARPACK path (num_bands) does "
                "not compute; leave num_bands out")
    if num_bands is not None:
      # ARPACK's eigs finds at most N-2 eigenpairs of an N x N matrix
      # (it needs k<N-1), so N-1 bands used to get past this and raise
      # TypeError there
      if num_bands>=(h.intra.shape[0]-1): num_bands=None
    if isinstance(operator,(list,)):
        operator = [h.get_operator(o) for o in operator]
    elif operator is not None: operator = h.get_operator(operator)
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
        if num_bands is None: # the dense eig, see orthonormal_levels
            ws = orthonormal_levels(algebra.todense(hk),es,ws)
        if biorthogonal: # <L_n|A|R_n>, the left vectors the rows of R^-1
            rinv = algebra.inv(algebra.todense(ws)) # biorthonormal to R
            ops = operator if isinstance(operator,(list,)) else [operator]
            wlr = [biorthogonal_weights(A,ws,rinv,kpath[k]) for A in ops]
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
        for (n,(e,w)) in enumerate(zip(es,ws)):  # loop over waves
            if callable(ewindow):
                if not ewindow(e): continue # skip iteration
            if biorthogonal: waws = [wA[n] for wA in wlr]
            elif isinstance(operator, (list,)): # input is a list
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
    else:
      raise ValueError("unknown eigmode; the accepted ones are 'complex', "
              "'real' and 'imag'")
    if write: # write the bands, as the Hermitian get_bands_nd does
      out = esk.real # kpoint index, energy and operator expectation values
      if eigmode=="complex": # keep the imaginary part, in its own column
          out = np.concatenate([out[0:2],[esk[1].imag],out[2:]])
      with open(output_file,"w") as f: np.savetxt(f,out.T) # write in file
    return esk




def biorthogonal_weights(A,R,rinv,k):
    """The biorthogonal weights (R^-1 A R)_nn of the operator A on the
    eigenstates whose right eigenvectors are the columns of R, rinv being
    R^-1, whose rows are the left eigenvectors normalized to <L_n|R_n>=1"""
    if isinstance(A,operators.Operator): AR = A(R,k=k) # A on every column
    elif callable(A): AR = A(R,k=k)
    else: AR = A@R # a matrix
    AR = np.asarray(algebra.todense(AR))
    return np.sum(np.asarray(rinv)*AR.T,axis=1) # the diagonal of R^-1 A R



def orthonormal_levels(m,es,vs,tol=1e-10):
    """Right eigenvectors that are orthonormal inside every degenerate
    level of m that has a basis of them.

    scipy's eig returns some basis of a degenerate eigenspace, not an
    orthonormal one, so that the right eigenvector weights <R|A|R>/<R|R>
    of a level added up to more or less than its weight, Tr of A on the
    eigenspace: in the Hermitian limit a level of weight 4 came out 4.17.
    Any combination of the vectors of a level is an eigenvector too, so
    they are replaced by an orthonormal basis of their span, which leaves
    the weight of the level independent of the basis. This is done only
    where the new vectors are still eigenvectors: at an exceptional point
    the vectors that eig returns for the coalescing states are nearly
    parallel, and an orthonormal basis of their span is not made of
    eigenvectors, so such a level is left as it is"""
    es = np.asarray(es) ; vs = np.array(vs)
    scale = max(1.,np.max(np.abs(es))) if len(es)>0 else 1.
    done = np.zeros(len(es),dtype=bool)
    for i in range(len(es)):
        if done[i]: continue
        level = np.where(np.abs(es-es[i])<tol*scale)[0] # this level
        done[level] = True
        if len(level)<2: continue # nothing to orthonormalize
        q = np.linalg.qr(vs[:,level])[0] # orthonormal basis of the span
        e0 = np.mean(es[level])
        if np.max(np.abs(m@q - e0*q))<1e-8*scale: vs[:,level] = q
    return vs
