# Intrinsic anomalous Hall conductivity of a 2D crystal, metal or insulator,
# from the Fermi-sea integral of the Berry curvature,
#   sigma_xy = (e^2/h) (1/2 pi) sum_n int d^2k f(E_n(k) - fermi) Omega_n(k),
# with the integral over the Brillouin zone.
#
# The Berry curvature of each band is the Kubo sum over the other bands,
#   Omega_n = -2 Im sum_{m not in the multiplet of n}
#             <n|v_x|m><m|v_y|n> / (E_n - E_m)^2,
# with the Cartesian velocities of conductivitytk/kubo.py, v = dH/dk + i[H,r],
# which carry the full bond vector (the atomic gauge). The gauge matters
# here, unlike for the Chern number: the curvature of the lattice gauge
# differs from the atomic one by the curl of <u_n|r|u_n>, a total derivative
# that integrates to zero over a filled band but not over a Fermi sea
# (measured on a strained honeycomb metal, 0.697 against 0.690, see
# future_development/triangle_anomalous_hall.md; C3 symmetry makes the
# difference vanish, so an unstrained honeycomb cannot see it). So
# the Wilson-loop plaquettes of topology.berry_curvature, which live in the
# lattice gauge and are set to zero where the number of occupied states
# changes inside the plaquette, are not used for a metal.
#
# Two integrations of the Fermi sea, selected by integration=:
#
# "grid": the uniform mean over the nk x nk mesh of f(E_n - fermi) Omega_n.
# In an insulator this converges exponentially with nk. In a metal the
# Fermi step cuts the mesh, and the sum only converges once the energy
# spacing of the mesh, of order v_F/nk, is below T, with an error that
# oscillates with nk.
#
# "triangle": E_n(k) and Omega_n(k) are interpolated linearly in each
# triangle of the same mesh (two per square), and theta(fermi - E) Omega is
# integrated exactly on each triangle. Since Omega is linear, this is a
# scalar weight per vertex, the weights of the linear tetrahedron method in
# 2D (dmtk/simplexdm.py, which also gives the finite-T weights as the T=0
# ones convolved with -f'(x), and the Bloechl term). A fully occupied
# triangle gives every vertex the uniform weight 1/nk^2, so in an insulator
# the triangle and the grid are the same number. In a metal the error is of
# second order in 1/nk at any T, T=0 included.
#
# Degeneracies. The weights are averaged over the degenerate multiplets at
# each k, and the pairs inside a multiplet are left out of Omega, with the
# same grouping (consecutive energies closer than the tolerance). The terms
# of a pair n, m enter Omega_n and Omega_m with opposite signs, so with equal
# weights they cancel exactly and the multiplet contributes its non-Abelian
# trace, which is gauge invariant. A crossing or an avoided crossing that is
# narrower than the mesh spacing still gives a peaked Omega_n of opposite
# sign in the two bands, and when the Fermi level cuts it the linear
# interpolation of the ordered bands makes an error of order one in a window
# of width of order v/nk around it.
#
# Units and sign. The result is in units of e^2/h, and in a gap it is the
# Chern number of h.get_chern() for a right-handed lattice basis (a1 x a2
# along +z). The curvature here is Cartesian while get_chern works in the
# reduced coordinates, so for a left-handed basis we expect the two to
# differ by a sign (reasoned, not tested). conductivity.py and
# conductivitytk/kpmkubo.py give sigma_xy in units of e^2/hbar with
# sigma_xy 2 pi = -C, so the same physical quantity there is
# -anomalous_hall_conductivity/(2 pi).
import numpy as np
from numba import jit, prange

from .. import parallel


default_nk = 100 # k-points per direction when the caller gives none
bloechl_default = True # the Bloechl term of the triangle weights, see
                       # future_development/triangle_anomalous_hall.md
_max_batch_bytes = 2e8 # memory of the velocity matrices of one batch


@jit(nopython=True,parallel=True,cache=True)
def _band_curvature_jit(es,vx,vy,tol,out):
    """Kubo Berry curvature of every band, out[k,n], leaving out the pairs
    inside a degenerate multiplet (consecutive energies closer than tol)"""
    nk = es.shape[0]; nb = es.shape[1]
    for ik in prange(nk):
        lab = np.zeros(nb,dtype=np.int64)
        for n in range(1,nb): # multiplet labels, chained as the weights are
            if es[ik,n]-es[ik,n-1]<tol: lab[n] = lab[n-1]
            else: lab[n] = lab[n-1]+1
        for n in range(nb):
            acc = 0.
            for m in range(nb):
                if lab[m]==lab[n]: continue
                d = es[ik,n]-es[ik,m]
                acc += (vx[ik,n,m]*vy[ik,m,n]).imag/(d*d)
            out[ik,n] = -2.*acc


def check_hamiltonian(h):
    """Refuse what the Fermi-sea Hall conductivity does not cover"""
    if h.dimensionality!=2:
        raise ValueError("the anomalous Hall conductivity needs a 2D "
                "periodic Hamiltonian, got dimensionality "
                +str(h.dimensionality))
    if h.has_eh:
        raise NotImplementedError("the anomalous Hall conductivity is only "
                "implemented for a normal-state (non-Nambu) Hamiltonian")
    from ..dmtk.simplexdm import check_hamiltonian as check_simplex
    check_simplex(h,"the anomalous Hall conductivity")


def band_curvature_mesh(h,nk=default_nk,degeneracy_tol=1e-6):
    """Band energies es[k,n] and Kubo Berry curvatures Omega[k,n] (Cartesian,
    per unit area of k-space) on the uniform nk x nk mesh, row-major as
    klist.kmesh, together with the cell area and the absolute degeneracy
    tolerance, degeneracy_tol times the hopping scale of h"""
    check_hamiltonian(h)
    from ..conductivitytk import kubo
    from ..dostk.simplexdos import uniform_mesh
    setup = kubo._setup(h)
    setup = (setup,kubo._hopping_arrays(setup[0]))
    cellvol,scale = setup[0][5],setup[0][6]
    tol = degeneracy_tol*scale
    ks = uniform_mesh(2,nk)
    n = setup[0][0].intra.shape[0]
    batch = int(max(64,min(2**14,_max_batch_bytes/(48.*n*n))))
    es = np.zeros((len(ks),n))
    om = np.zeros((len(ks),n))
    parallel.set_num_threads()
    for i0 in range(0,len(ks),batch):
        eb,vb = kubo._bands_and_velocities(h,ks[i0:i0+batch],setup=setup)[:2]
        ob = np.zeros(eb.shape)
        _band_curvature_jit(eb,np.ascontiguousarray(vb[:,0]),
                np.ascontiguousarray(vb[:,1]),tol,ob)
        es[i0:i0+len(eb)] = eb
        om[i0:i0+len(eb)] = ob
    return es,om,cellvol,tol


def _grid_weights(es,nk,fermi,T,tol,bloechl):
    """Fermi occupations on the uniform mesh, divided by the number of
    k-points (bloechl does not apply)"""
    x = es-fermi
    if T<=0.: f = np.where(x<0.,1.,0.)
    else: f = 1./(1.+np.exp(np.clip(x/T,-500.,500.)))
    return f/nk**2


def _triangle_weights(es,nk,fermi,T,tol,bloechl):
    """Linear triangle weights, averaged over the degenerate multiplets"""
    from ..dmtk.simplexdm import simplex_weights, _multiplet_average_real
    w = simplex_weights(2,nk,es,mu=fermi,T=T,bloechl=bloechl)
    _multiplet_average_real(es,w,tol)
    return w


# integration= of anomalous_hall_conductivity -> occupation weights w[k,n]
_integrations = {
    "grid": _grid_weights,
    "triangle": _triangle_weights,
    }


def get_integration_names():
    """Accepted values of integration= in anomalous_hall_conductivity"""
    return list(_integrations)


def anomalous_hall_conductivity(h,fermi=0.,T=0.,nk=default_nk,
        integration="triangle",bloechl=None,degeneracy_tol=1e-6):
    """Intrinsic anomalous Hall conductivity of a 2D Hamiltonian, the
    Fermi-sea integral of the Berry curvature,
      sigma_xy = (1/2 pi) sum_n int d^2k f(E_n(k) - fermi) Omega_n(k),
    in units of e^2/h. In a gap it is the Chern number of h.get_chern()
    (right-handed lattice basis); in a metal it is not quantized. The
    conductivity.py and kpmkubo convention, e^2/hbar with
    sigma_xy 2 pi = -C, is minus this divided by 2 pi.

    Parameters
    ----------
    fermi : float or array
      Fermi level(s); the mesh is diagonalized once for all of them
    T : float
      temperature of the Fermi function, T=0 is the step
    nk : int
      k-points per direction of the uniform mesh
    integration : str
      one of get_integration_names(): "triangle" (default) integrates the
      Fermi step exactly on the triangles of the mesh, with E_n and Omega_n
      linear in each, so a metal converges as 1/nk^2 at any T; "grid" is
      the uniform mean of the Fermi-weighted curvature, which in a metal
      needs a mesh spacing below T. In a gap both give the same number
    bloechl : bool or None
      add the Bloechl term to the triangle weights, None takes
      bloechl_default (True)
    degeneracy_tol : float
      relative tolerance (in units of the hopping scale) below which two
      bands at the same k count as one multiplet

    Returns
    -------
    a float, or an array with one value per Fermi level
    """
    if integration not in _integrations:
        raise ValueError("unknown integration "+repr(integration)+" for the "
                "anomalous Hall conductivity; it must be one of "
                +str(get_integration_names()))
    if bloechl is None: bloechl = bloechl_default
    T = 0. if T is None else float(abs(T))
    es,om,cellvol,tol = band_curvature_mesh(h,nk=nk,
            degeneracy_tol=degeneracy_tol)
    wf = _integrations[integration]
    fermis = np.atleast_1d(np.array(fermi,dtype=np.float64))
    out = np.array([np.sum(wf(es,nk,float(mu),T,tol,bloechl)*om)
        for mu in fermis])
    out = out*2.*np.pi/cellvol # (1/2 pi) int d^2k = (2 pi/A_cell) mean_k
    if np.ndim(fermi)==0: return float(out[0])
    return out
