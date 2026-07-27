import numpy as np

# density (charge) RPA response functions, the density-density analog of
# spinchi.py's spin-channel functions. The charge channel needs no special
# operator: chi_AB_RPA's default (A=B=None -> identity, i.e. the per-site
# total-number operator via chiAB's own site projectors, which sum both
# spins -- see operators.index) already is the density-density response,
# an N-dimensional (one entry per site) matrix -- NOT the 2N spin-orbital
# one selfconsistency.spinspin._build_density_v returns for Hartree-Fock
# decoupling, so that helper cannot be reused as-is here; _density_v below
# builds the N-dimensional interaction the charge channel actually needs.


def _density_v(h,V1=0.,V2=0.,V3=0.,U=0.,Vr=None,nd=None):
    """Build the per-site (N-dimensional, not spin-doubled) density-density
    interaction matrix for the charge/density RPA channel.

    V1/V2/V3/Vr (couplings between *different* sites) carry over directly
    and unambiguously: (1/2) V_ij n_i n_j (i!=j, n_i the total occupation
    at site i) is already a two-site term with no spin/charge mixing, so
    its charge-channel coefficient is exactly V_ij -- same neighbor-shell
    construction as selfconsistency.spinspin._build_v/_build_density_v,
    just without their spin-doubling step.

    The onsite U term needs care: U n_up n_down = (U/4)n^2 - (U/4)m_z^2
    (n = n_up+n_down, m_z = n_up-n_down) splits into a +U/4 coefficient of
    n^2 and a -U/4 coefficient of m_z^2. Matching the (a/2)n^2 form the RPA
    kernel 1-a*chi0 assumes gives a_charge = U/2 for the charge channel --
    cross-checked for consistency against the -2U spin-channel coefficient
    chitk.spinchi._full_spin_U already uses for the same U: -(U/4)*m_z^2 =
    -(U/4)*4*Sz^2 = -U*Sz^2, matched to (a/2)Sz^2 gives a_spin/2 = -U ->
    a_spin = -2U -- same physical U, same decomposition, opposite channel."""
    from .. import specialhopping
    if nd is None: nd = h.geometry.neighbor_distances()
    mgenerator = specialhopping.distance_hopping_matrix(
            [V1/2.,V2/2.,V3/2.], nd[0:3])
    hv = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
            mgenerator=mgenerator)
    if Vr is not None:
        hv1 = h.geometry.get_hamiltonian(has_spin=False, is_multicell=True,
                tij=Vr)
        hv = hv + hv1
    v = hv.get_hopping_dict()
    from ..selfconsistency.densitydensity import obj2geometryarray
    Ua = obj2geometryarray(U,h.geometry)
    n = len(h.geometry.r)
    for i in range(n):
        v[(0,0,0)][i,i] = v[(0,0,0)][i,i] + Ua[i]/2.
    return v


def densitychi_RPA(h,V1=0.,V2=0.,V3=0.,U=0.,Vr=None,**kwargs):
    """Return the density (charge) RPA response function for a
    V1/V2/V3 neighbor-shell (+ onsite U, + optional general Vr(r))
    density-density interaction -- same V1/V2/V3/U/Vr convention as
    selfconsistency.densitydensity.Vinteraction/VJinteraction. Unlike the
    spin channel (spinchi_full/magnon_bands), this does not need a
    converged mean-field Hamiltonian first: the interaction is fully
    determined by V1/V2/V3/U/Vr, so it can dress the bare susceptibility of
    any Hamiltonian directly (h can also be an already-converged one, e.g.
    from VJinteraction, if you want the RPA response about that reference
    state instead)."""
    from .rpa import chi_AB_RPA
    h1 = h.get_multicell().get_dense()
    v = _density_v(h1,V1,V2,V3,U,Vr)
    return chi_AB_RPA(h1,V=v,**kwargs)


def plasmon_bands(h,V1=0.,V2=0.,V3=0.,U=0.,Vr=None,qpath=None,nq=20,**kwargs):
    """Return the plasmon/charge-order bands: the poles of the density RPA
    kernel 1 - V(q)*chi0(q,omega) for a V1/V2/V3/U/Vr neighbor-shell
    density-density interaction, scanned along a q-path -- the charge-
    channel analog of spinchi.magnon_bands. The interaction is taken
    directly as parameters (not read back from H.V the way magnon_bands
    reads the spin-exchange interaction from a converged SCF): it is fully
    determined by V1/V2/V3/U/Vr, so it is simplest to just rebuild it once
    and reuse it at every q, whether h is a bare or an already-converged
    Hamiltonian.

    Returns (qs,ws,gammas): qs is the integer index of the q-point along
    the path (the same convention get_bands/magnon_bands use), ws the pole
    frequency and gammas its residual imaginary part -- signed, so judge
    how sharp/well-defined a mode is by abs(gammas), not gammas directly
    (see rpa.py's _poles_from_chi_matrix docstring). Different q-points can
    have different numbers of poles, so all three are flat 1D arrays."""
    from .rpa import rpa_kernel_poles
    from .. import parallel
    h1 = h.get_multicell().get_dense()
    v = _density_v(h1,V1,V2,V3,U,Vr)
    qpath = h1.geometry.get_kpath(qpath,nk=nq) # generate the q-path
    def f(q):
        return rpa_kernel_poles(h1,V=v,q=q,**kwargs)
    outs = parallel.pcall(f,qpath) # compute the poles at every q
    qs,ws,gammas = [],[],[] # flat storage
    for iq,poles in enumerate(outs): # loop over q-points
        for (w,g) in poles: # loop over poles found at this q
            qs.append(iq)
            ws.append(w)
            gammas.append(g)
    return np.array(qs),np.array(ws),np.array(gammas)
