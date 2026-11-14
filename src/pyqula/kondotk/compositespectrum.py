"""Matrix spectral function of the conduction electron and the composite
fermion of a Kondo lattice mean field, resolved in momentum.

Chen, Stoudenmire, Komijani and Coleman, arXiv:2302.09701, write the Kondo
lattice as

    H = sum_k eps_k c^dag_k c_k + J' sum_j (c^dag_j sigma c_j).S_j ,

and define the composite fermion, an electron together with a spin flip of
the moment of the same site,

    F_j,a = (2/3) sum_b (sigma.S_j)_ab c_j,b .

It is the operator that the Kondo term couples to the electron, since
i d/dt c_j = [c_j, H] gives eps c_j + (3J'/2) F_j. The paper combines the
two operators in the spinor psi = (c, F) and computes the retarded matrix
Green function (its Eq. green)

    G(k,t) = -i theta(t) <{psi_k(t), psi^dag_k(0)}> = [[G_cc, G_cF],
                                                        [G_Fc, G_FF]] ,

with matrix product states, and compares its spectral functions
A_cc = -Im G_cc/pi and A_FF = -Im G_FF/pi with the auxiliary fermion mean
field. The equation of motion of c makes three relations exact (its Eq.
relations and Eq. matrixG), with g_c = 1/(z - eps_k):

    G_cc = g_c + g_c (3J'/2) G_FF (3J'/2) g_c ,
    G_cF = G_Fc = g_c (3J'/2) G_FF ,
    g_F = 1/[G^-1]_FF ,

where g_F is the propagator of the composite fermion without its
hybridization with the conduction electrons.

In the mean field the moment is S_j = f^dag_j sigma f_j / 2 and the Kondo
term becomes V_j f^dag_j c_j + h.c., so the same equation of motion reads
i d/dt c_j = eps c_j + conj(V_j) f_j. The composite fermion is therefore
the f fermion times an amplitude,

    F_j = z_j f_j ,     z_j = 2 conj(V_j)/(3J') = 4 conj(V_j)/(3J) ,

with J = 2J' the Coqblin-Schrieffer coupling that the self-consistent loop
of this library takes (the quartic term of J'(c^dag sigma c).S is
-J' X^dag X with X = sum_a c^dag_a f_a, and the loop decouples
-(J/2) X^dag X). The matrix Green function of the mean field is then

    G_cc = G^mf_cc ,  G_cF = conj(z) G^mf_cf ,  G_FF = |z|^2 G^mf_ff ,

and it obeys the three relations above exactly.

Two things are known about the mean field and are left to the caller as
keywords:

- At strong coupling the loop gives V = J' and the exact energy to add or
  remove one electron is 3J'/2, so the mean field has the right two bands
  with a hybridization that is 2/3 of the exact one, and a weight
  |z|^2 = 4/9 instead of 1. The paper rescales V by 3/2 by hand when it
  compares the bands with the matrix product states (its section
  "Comparison with strong coupling and mean-field"); this is
  hybridization_scale=1.5 here. The rescale keeps the f levels, so the
  f occupation stays at one only where the spectrum is particle-hole
  symmetric.
- The paper identifies F with f. With J=None the amplitude is z=1, which
  is that identification. With J given, z is the amplitude above.

The spectral functions are built from the eigenstates of the mean-field
Hamiltonian, one diagonalization per k-point, with the weights of every
state on the two orbitals of each Kondo pair, as kondotk.cotunneling does
for the sum over k. So no matrix is inverted per energy and a cell of a
few hundred Kondo pairs is affordable. Energies are measured from the
chemical potential of the self-consistent loop.

Nothing here is specific to the chain of the paper. The Hamiltonian can
have any dimensionality, 0D (a finite cluster, one k-point), 1D, 2D or 3D,
and any number of Kondo pairs per unit cell. With several pairs the matrix
Green function is a matrix in the pairs too, G^-1 = z - H(k) in the basis
of every c and every F of the cell, and g_F is the inverse of its FF
block, the Schur complement G_FF - G_Fc G_cc^-1 G_cF. The routine returns
the elements that are diagonal in the pair."""
import numpy as np

from .. import algebra
from . import pairs as pairstk


def composite_amplitude(h, pairs, J):
    """Amplitude z_j of the f fermion in the composite fermion F_j = z_j f_j
    of every Kondo pair: 1 with J=None, and 4 conj(V_j)/(3J_j) with J_j
    the Coqblin-Schrieffer coupling of the onsite bond of the moment and
    V_j its hybridization, the one that h holds. J is a number, or the
    function J(rc, rm) of the self-consistent loop, read at rc = rm. The
    composite fermion is the one of the site, (2/3)(sigma.S_j)c_j, so only
    the onsite bond enters even when the moment is coupled to other sites
    too, and a moment without an onsite coupling has no amplitude here."""
    if J is None: return np.ones(len(pairs), dtype=np.complex128)
    Jm = pairstk.onsite_coupling(J, h, pairs)
    if not callable(J) and J == 0.0:
        raise ValueError("J=0 makes the amplitude 4V/(3J) of the composite "
                "fermion undefined; pass the Coqblin-Schrieffer coupling "
                "used in the self-consistent loop, or J=None for an "
                "amplitude of 1")
    pairstk.require_onsite_coupling(Jm, "the composite fermion")
    return 4.*np.conjugate(pairstk.pair_hybridization(h, pairs))/(3.*Jm)


def _default_energies(es, delta, ne, emax):
    """Energy grid that covers the bands of the first k-point"""
    if emax is None: emax = 1.2*np.max(np.abs(es)) + 10*delta
    return np.linspace(-emax, emax, ne)


def composite_spectral_function(h, kpath=None, energies=None, delta=2e-2,
        J=None, hybridization_scale=1.0, site=None, green=False, nk=100,
        ne=400, emax=None):
    """Spectral functions of the conduction electron c and the composite
    fermion F of a Kondo lattice mean field along a k-path, and the
    propagator g_F of the composite fermion without its hybridization
    (arXiv:2302.09701, see the module docstring).

    h: Hamiltonian of a Kondo lattice mean field, the one returned by
        KondoLatticeHamiltonian.get_mean_field_hamiltonian (dense, spinful).
    kpath: k-points in units of the reciprocal lattice vectors, or labels of
        high-symmetry points, as in get_bands. Default: the path of the
        geometry with nk points. A finite (0D) system has no momentum: the
        result has one k-point, and its spectral functions are the local
        ones of each Kondo pair.
    energies: array of energies, measured from the chemical potential.
        Default: ne points from -emax to emax, with emax given or 1.2 times
        the largest |eigenvalue| at the first k-point plus 10*delta.
    delta: Lorentzian broadening of the levels, in the units of h.
    J: None, and the composite fermion is the f fermion (F = f, the
        identification of the paper). Or the Coqblin-Schrieffer coupling of
        the self-consistent loop, twice the J of the paper, and F is
        (4 conj(V)/(3J)) f, the amplitude that the equation of motion of c
        gives, so that A_FF integrates to |4V/(3J)|^2. With a coupling to
        other sites, pass the same function J(rc, rm) as to the loop: V
        and J are then those of the onsite bond of the moment, and a moment
        without one raises.
    hybridization_scale: factor that multiplies the hybridization V of the
        mean field before the spectrum is computed. The paper uses 1.5 to
        compare with the matrix product states, since the mean field gives
        2/3 of the exact hybridization at strong coupling.
    site: Kondo pair index, or list of them. Default: the average over the
        pairs of the unit cell.
    green: if True, return the retarded Green functions G(k, w + i delta)
        themselves, complex, instead of their spectral functions, together
        with "Fc", so that the 2 x 2 matrix can be inverted as in the paper.

    Returns (ks, energies, out). ks are the k-points, an array of shape
    (number of k-points, 3). out is a dictionary of arrays of shape (number
    of k-points, number of energies), with one more leading index when site
    is a list:

        "cc": A_cc = -Im G_cc/pi, the conduction electron,
        "FF": A_FF = -Im G_FF/pi, the composite fermion,
        "cF": the spectral function of G_cF, i(G_cF - conj(G_Fc))/(2 pi),
              which is real when V is real,
        "gF": g_F(k, w + i delta), complex, the composite fermion without
              the hybridization, 1/[G^-1]_FF.

    Each one is the average over the two spin components, so that A_cc
    integrates to one, and diagonal in the Kondo pair. One diagonalization
    per k-point is done, of the block of one spin when the Hamiltonian is
    the same for the two spins and does not mix them."""
    what = "the composite fermion spectral function"
    pairs = pairstk.get_pairs(h, what)
    sites = pairstk.select_sites(site, len(pairs))
    if delta <= 0.0:
        raise ValueError("delta must be positive, the broadening of the "
                "levels; got "+str(delta))
    h = pairstk.scale_hybridization(h, hybridization_scale, pairs)
    z = composite_amplitude(h, pairs, J)[sites]
    ci = np.array([pairs[s][0] for s in sites])
    fi = np.array([pairs[s][1] for s in sites])
    # spin orbitals of every f site of the cell, and where the selected
    # pairs are among them
    fall = np.array([p[1] for p in pairs])
    forb = np.concatenate([[2*f, 2*f+1] for f in fall])
    fpos = np.array([list(fall).index(f) for f in fi])
    if h.dimensionality == 0: # a finite system has no momentum
        kpath = np.zeros((1, 3))
    else:
        kpath = np.asarray(h.geometry.get_kpath(kpath, nk=nk, write=False),
                dtype=float)
    hkgen = h.get_hk_gen()
    nsel = len(sites)
    degenerate = pairstk.is_spin_degenerate(h) # one spin block is enough
    out = None
    for ik, k in enumerate(kpath):
        hk = np.asarray(algebra.todense(hkgen(k)))
        # weights of every eigenstate on c and f of each pair, over spin,
        # from one diagonalization
        es, wcc, wff, wcf = pairstk.pair_weights(hk, ci, fi, degenerate)
        if out is None:
            if energies is None:
                energies = _default_energies(es, delta, ne, emax)
            energies = np.atleast_1d(np.asarray(energies, dtype=float))
            shape = (nsel, len(kpath), len(energies))
            out = {key: np.zeros(shape, dtype=np.complex128)
                    for key in ("cc", "FF", "cF", "Fc", "gF")}
        # poles of the two spin components, averaged
        pole = 0.5/(energies[:, None] + 1j*delta - es[None, :])
        out["cc"][:, ik, :] = wcc@pole.T
        out["FF"][:, ik, :] = (np.abs(z)**2)[:, None]*(wff@pole.T)
        out["cF"][:, ik, :] = np.conjugate(z)[:, None]*(wcf@pole.T)
        out["Fc"][:, ik, :] = z[:, None]*(np.conjugate(wcf)@pole.T)
        # the f block alone: the composite fermion without hybridization
        ef, comps, mult = pairstk.spin_components(hk[np.ix_(forb, forb)],
                degenerate)
        wf = mult*sum(np.abs(u[fpos, :])**2 for u in comps)
        pole = 0.5/(energies[:, None] + 1j*delta - ef[None, :])
        out["gF"][:, ik, :] = (np.abs(z)**2)[:, None]*(wf@pole.T)
    if not green: # the spectral functions
        Fc = out.pop("Fc")
        out["cF"] = 1j*(out["cF"] - np.conjugate(Fc))/(2*np.pi)
        out["cc"] = -out["cc"].imag/np.pi
        out["FF"] = -out["FF"].imag/np.pi
    if site is None: # the average over the pairs of the cell
        out = {key: np.mean(val, axis=0) for (key, val) in out.items()}
    elif np.ndim(site) == 0: # one pair, one array
        out = {key: val[0] for (key, val) in out.items()}
    return kpath, energies, out
