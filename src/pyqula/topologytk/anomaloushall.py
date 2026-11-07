# Intrinsic anomalous Hall conductivity of a 2D or 3D crystal, metal or
# insulator, from the Fermi-sea integral of the Berry curvature,
#   sigma_xy = (e^2/h) (1/2 pi) sum_n int d^2k f(E_n(k) - fermi) Omega_n(k),
# with the integral over the Brillouin zone. In 3D the same integral runs
# over the 3D zone with an extra 1/(2 pi),
#   sigma_ab = (e^2/h) (1/(2 pi)^2) sum_n int d^3k f Omega^ab_n,
# a conductivity per unit length: for decoupled layers of Chern number C
# and spacing c it is C/c. In both cases the mean over a uniform mesh gives
# sigma = (2 pi/V_cell) mean_k sum_n f Omega_n, with V_cell the area in 2D
# and the volume in 3D, so one code serves both.
#
# The Berry curvature of each band is the Kubo sum over the other bands,
#   Omega^ab_n = -2 Im sum_{m not in the multiplet of n}
#             <n|v_a|m><m|v_b|n> / (E_n - E_m)^2,
# with (a,b) = (x,y) by default and, in 3D, the plane chosen by direction=
# ("yz" is Omega_x, "zx" is Omega_y, a reversed pair flips the sign),
# and with the Cartesian velocities of conductivitytk/kubo.py, v = dH/dk + i[H,r],
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
# second order in 1/nk at any T, T=0 included. In 3D the same holds with
# the six tetrahedra per cube of dostk/simplexdos.py, which share the
# diagonal from (0,0,0) to (1,1,1) of the reduced mesh. For decoupled
# layers stacked along a3, the three tetrahedra above a triangle fill the
# prism on it, so the 3D weights are the 2D ones of triangles cut along
# (0,0)-(1,1), which is the 2D mesh of the same layer in the basis
# (a1,-a2): there the two agree to rounding, and with the basis (a1,a2)
# at second order in 1/nk. The Bloechl term breaks this, since its
# coefficient depends on the dimension (2e-4 for a Haldane metal at nk=24).
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
    if h.dimensionality not in (2,3):
        raise ValueError("the anomalous Hall conductivity needs a 2D or 3D "
                "periodic Hamiltonian, got dimensionality "
                +str(h.dimensionality))
    if h.has_eh:
        raise NotImplementedError("the anomalous Hall conductivity is only "
                "implemented for a normal-state (non-Nambu) Hamiltonian")
    from ..dmtk.simplexdm import check_hamiltonian as check_simplex
    check_simplex(h,"the anomalous Hall conductivity")


# direction= of anomalous_hall_conductivity -> the Cartesian pair (a,b) of
# sigma_ab, whose curvature is built from v_a and v_b
_directions = {"xy":(0,1),"yz":(1,2),"zx":(2,0),
               "yx":(1,0),"zy":(2,1),"xz":(0,2)}


def get_direction_names():
    """Accepted values of direction= in anomalous_hall_conductivity"""
    return list(_directions)


def _axes(direction):
    if direction not in _directions:
        raise ValueError("unknown direction "+repr(direction)+" of the "
                "anomalous Hall conductivity; it must be one of "
                +str(get_direction_names()))
    return _directions[direction]


def band_curvature_mesh(h,nk=default_nk,degeneracy_tol=1e-6,direction="xy"):
    """Band energies es[k,n] and Kubo Berry curvatures Omega^ab[k,n]
    (Cartesian, per unit area of k-space, (a,b) set by direction) on the
    uniform nk^dim mesh, row-major as klist.kmesh, together with the cell
    area (2D) or volume (3D) and the absolute degeneracy tolerance,
    degeneracy_tol times the hopping scale of h"""
    check_hamiltonian(h)
    a,b = _axes(direction)
    dim = h.dimensionality
    from ..conductivitytk import kubo
    from ..dostk.simplexdos import uniform_mesh
    setup = kubo._setup(h)
    setup = (setup,kubo._hopping_arrays(setup[0]))
    cellvol,scale = setup[0][5],setup[0][6]
    tol = degeneracy_tol*scale
    ks = uniform_mesh(dim,nk)
    n = setup[0][0].intra.shape[0]
    batch = int(max(64,min(2**14,_max_batch_bytes/(48.*n*n))))
    es = np.zeros((len(ks),n))
    om = np.zeros((len(ks),n))
    parallel.set_num_threads()
    for i0 in range(0,len(ks),batch):
        eb,vb = kubo._bands_and_velocities(h,ks[i0:i0+batch],setup=setup)[:2]
        ob = np.zeros(eb.shape)
        _band_curvature_jit(eb,np.ascontiguousarray(vb[:,a]),
                np.ascontiguousarray(vb[:,b]),tol,ob)
        es[i0:i0+len(eb)] = eb
        om[i0:i0+len(eb)] = ob
    return es,om,cellvol,tol


def _grid_weights(dim,es,nk,fermi,T,tol,bloechl):
    """Fermi occupations on the uniform mesh, divided by the number of
    k-points (bloechl does not apply)"""
    x = es-fermi
    if T<=0.: f = np.where(x<0.,1.,0.)
    else: f = 1./(1.+np.exp(np.clip(x/T,-500.,500.)))
    return f/nk**dim


def _triangle_weights(dim,es,nk,fermi,T,tol,bloechl):
    """Linear triangle (2D) or tetrahedron (3D) weights, averaged over the
    degenerate multiplets"""
    from ..dmtk.simplexdm import simplex_weights, _multiplet_average_real
    w = simplex_weights(dim,nk,es,mu=fermi,T=T,bloechl=bloechl)
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
        integration="triangle",bloechl=None,degeneracy_tol=1e-6,
        direction="xy"):
    """Intrinsic anomalous Hall conductivity of a 2D or 3D Hamiltonian, the
    Fermi-sea integral of the Berry curvature,
      sigma_xy = (1/2 pi) sum_n int d^2k f(E_n(k) - fermi) Omega_n(k)
    in 2D, in units of e^2/h. In a gap it is the Chern number of
    h.get_chern() (right-handed lattice basis); in a metal it is not
    quantized. The conductivity.py and kpmkubo convention, e^2/hbar with
    sigma_xy 2 pi = -C, is minus this divided by 2 pi.

    In 3D it is
      sigma_ab = (1/(2 pi)^2) sum_n int d^3k f(E_n(k) - fermi) Omega^ab_n(k),
    a conductivity, in units of e^2/(h a) with a the length unit of the
    geometry (the unit of g.r and of the lattice vectors), with (a,b) the
    plane chosen by direction. For decoupled Chern layers stacked with a
    spacing c it is C/c. In a 3D insulator it is quantized in the same
    way, a component of G/(2 pi) with G a reciprocal lattice vector.

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
    direction : str
      one of get_direction_names(), the Cartesian pair ab of sigma_ab:
      "xy" (default) is the curvature Omega_z, "yz" is Omega_x and "zx" is
      Omega_y, and a reversed pair ("yx") flips the sign. In 2D only "xy"
      and "yx" are accepted, since there is no current along z

    Returns
    -------
    a float, or an array with one value per Fermi level
    """
    if integration not in _integrations:
        raise ValueError("unknown integration "+repr(integration)+" for the "
                "anomalous Hall conductivity; it must be one of "
                +str(get_integration_names()))
    _axes(direction) # refuse an unknown direction first
    if h.dimensionality==2 and direction not in ("xy","yx"):
        raise ValueError("a 2D Hamiltonian has a Hall conductivity only in "
                "the plane, direction='xy' or 'yx', got direction="
                +repr(direction))
    if bloechl is None: bloechl = bloechl_default
    T = 0. if T is None else float(abs(T))
    es,om,cellvol,tol = band_curvature_mesh(h,nk=nk,
            degeneracy_tol=degeneracy_tol,direction=direction)
    wf = _integrations[integration]
    dim = h.dimensionality
    fermis = np.atleast_1d(np.array(fermi,dtype=np.float64))
    out = np.array([np.sum(wf(dim,es,nk,float(mu),T,tol,bloechl)*om)
        for mu in fermis])
    # (1/2 pi) int d^2k = (2 pi/A_cell) mean_k in 2D, and
    # (1/(2 pi)^2) int d^3k = (2 pi/V_cell) mean_k in 3D
    out = out*2.*np.pi/cellvol
    if np.ndim(fermi)==0: return float(out[0])
    return out
