# compute vacuum expectation values

import numpy as np
from .operators import Operator

def get_dm_vev(H,A,**kwargs):
    """Compute a vacuum expectation value of two operators"""
    if H.dimensionality != 0:
        raise ValueError("the density-matrix expectation value is only "
                "implemented for 0d Hamiltonians")
    # every keyword (the temperature above all) used to be dropped here,
    # so the vev was the zero-temperature one whatever was asked for
    dm = H.get_density_matrix(**kwargs) # return the DM, as a matrix
    A = Operator(A) # convert to operator
    # transposed for the same reason as in spectrum.ev: full_dm's
    # convention is the transpose of the usual density matrix, so
    # contracting it directly gives <A*> instead of <A>
    return np.trace(A@np.transpose(dm)) # return the expectation value



def kresolved_orbital_vev(h,operator,nk=30,fermi=0.0,T=None,delta=None,
        batch_size=16):
    """Expectation value of an operator that is applied inside the sum
    over kpoints, one entry per orbital.

    spectrum.ev contracts the operator with densitymatrix.full_dm, which
    has already summed over the Brillouin zone, so an operator that acts
    with a different matrix at every kpoint -- the unfolding projector,
    for one -- cannot be used there at all, and one that is defined only
    by its action on a wavefunction has no matrix to contract in the
    first place. Here the operator is applied to each occupied eigenstate
    at the kpoint that state belongs to,

        out_a = (1/Nk) sum_k sum_n f(e_nk) conj(psi_nk,a) (A(k) psi_nk)_a

    with the same Fermi-Dirac occupation, the same energy smearing and
    the same normalization that full_dm uses, so this agrees with
    spectrum.ev whenever both can be applied. Sum the result to get the
    total expectation value, or pass it through h.full2profile to resolve
    it by site."""
    from .densitymatrix import delta_dm
    from .htk.eigenvectors import peigh_bloch
    if T is not None and delta is not None and T!=delta:
        raise TypeError("kresolved_orbital_vev got both T="+str(T)+" and "
                "delta="+str(delta)+", which are the same energy smearing "
                "under two names; pass only one")
    smearing = T if T is not None else (delta if delta is not None else delta_dm)
    if smearing==0.: smearing = 1e-15 # just very small, as full_dm does
    op = h.get_operator(operator) # resolve names and matrices alike
    if op is None:
        raise ValueError("kresolved_orbital_vev needs an operator; without "
                "one use the density matrix directly")
    hk = h.get_hk_gen() # Bloch Hamiltonian generator
    ks = np.array(h.geometry.get_kmesh(nk=nk)) # the same mesh full_dm uses
    out = np.zeros(h.intra.shape[0]) # one entry per orbital
    for i0 in range(0,len(ks),batch_size): # batches of kpoints
        kbatch = ks[i0:i0+batch_size]
        # diagonalize the batch in parallel across numba threads, as
        # densitymatrix.full_dm_accumulate does
        es_batch,vs_batch = peigh_bloch(hk,kbatch)
        for ik in range(len(kbatch)):
            x = (es_batch[ik]-fermi)/smearing
            # the Fermi function, written so that neither tail overflows:
            # 1/(1+exp(x)) is 0.5*(1-tanh(x/2)), and with the default
            # smearing of 1e-6 the exponent reaches several hundred
            occ = 0.5*(1.0-np.tanh(0.5*x)) # Fermi-Dirac occupation
            ws = vs_batch[ik] # columns are eigenvectors
            for n in range(len(occ)):
                # with the default smearing the occupation is 0 or 1 to
                # many digits, so the empty states are simply skipped
                if occ[n]<1e-12: continue
                w = ws[:,n] # this eigenstate
                u = op.m(w,k=kbatch[ik]) # the operator, at this kpoint
                out += occ[n]*(np.conjugate(w)*u).real
    return out/len(ks) # normalize by the number of kpoints
