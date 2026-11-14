"""Dynamical spin susceptibility of the local moments of a Kondo lattice
mean field, with the residual interaction of the fractionalized
quasiparticles in the random phase approximation.

Chen, Stoudenmire, Komijani and Coleman, arXiv:2302.09701, compare the spin
susceptibility of the moments of the Kondo chain (their Eq. eqchi_S),

    chi_S(q,w) = sum_nm int_0^inf dt e^{i[(w+i eta)t - q(n-m)]}
                 (-i) <[S^-_n(t), S^+_m(0)]> ,

computed with matrix product states, with the one of the auxiliary fermion
mean field. There the moment is S_j = f^dag_j sigma f_j / 2, and the
susceptibility is the convolution of two f propagators (their appendix
"Spin-susceptibility within mean-field theory"): a spin flip of the moment
is a particle and a hole of the hybridized bands, each with its weight on
the f orbital. For the chain at half filling, with the two bands
E_-(k) < 0 < E_+(k) and the weight sin^2(alpha_k) of the f orbital in the
upper one,

    chi(q,w) = 1/N sum_k cos^2(alpha_k) sin^2(alpha_k+q)
               [ 1/(w - (E_+(k+q) - E_-(k))) - 1/(w + (E_+(k+q) - E_-(k))) ] ,

a continuum that starts at the indirect gap for q=pi and at the direct gap
2V for q=0. Here it is computed for any Kondo lattice Hamiltonian as the
bubble of chitk.localspinchi, restricted to the spin operators of the f
sites, so that a spin splitting, a supercell or a hopping between the
moments are all included, in any dimensionality: 0D (a finite cluster,
where there is no momentum and q is ignored), 1D, 2D or 3D. With several
moments per unit cell the response and the residual interaction are
matrices between them, chi_RPA = chi (1 - U' chi)^-1.

The mean field leaves out the interaction between the quasiparticles. The
paper puts it back as a residual interaction U'(q) in the random phase
approximation (its section "Strong coupling and mean-field perspective"),

    chi_RPA(q,w) = 1/(1/chi(q,w) - U'(q)) ,

and finds that a constant U' < 0 gives the magnon below the continuum that
the matrix product states show at strong coupling (U' = -2t for J' = 2t),
and that a U'(q) that vanishes at small q is needed at weak coupling
(U' = -0.75t sin(q/2) for J' = 0.9t). U' is a parameter of the comparison,
not a result of the mean field. Since chi^{+-} is twice chi^{zz} in a state
with spin rotation symmetry, this is the dressing of the interaction
U'(q) S_q.S_-q, an exchange between the moments of 2U' in the form
(1/2) sum W_ij S_i.S_j. It is applied here in that rotationally invariant
form to the three spin components, so that it is defined in a Zeeman field
too.

Sign convention, the one of the spin responses of this library: the real
part of the static response is negative, and the spectral weight at
positive frequency is +Im chi/pi."""
import numpy as np

from ..chitk import localspinchi
from . import pairs as pairstk


def _residual_at_q(U, q, nf):
    """The residual interaction at q as an nf x nf matrix between the
    moments of the cell"""
    if callable(U): U = U(np.array(q, dtype=np.float64))
    U = np.asarray(U, dtype=np.complex128)
    if U.ndim == 0: return U*np.identity(nf, dtype=np.complex128)
    if U.shape != (nf, nf):
        raise ValueError("the residual interaction U must be a number or a "
                "matrix between the %d moments of the unit cell, or a "
                "function of q that returns one of them; got an array of "
                "shape %s" % (nf, U.shape))
    return U


def moment_spinchi(h, q=None, energies=np.linspace(0.0, 4.0, 200), U=None,
        nk=60, delta=0.05, T=None, hybridization_scale=1.0):
    """Transverse spin susceptibility of the local moments of a Kondo
    lattice mean field, the response in which S^+ creates the excitation
    (arXiv:2302.09701, see the module docstring). Between the moments i
    and j of the unit cell the bare one is

        chi_ij(q,w) = 1/Nk sum_k sum_mn (f_m - f_n)
            <m,k|S^+_i|n,k+q> <n,k+q|S^-_j|m,k> / (e_m - e_n - w + i delta) ,

    with m a state at k and n a state at k+q: S^+ acts on the moment of
    the first index and S^- on the one of the second.

    h: Hamiltonian of a Kondo lattice mean field, the one returned by
        KondoLatticeHamiltonian.get_mean_field_hamiltonian (spinful).
    q: momentum in units of the reciprocal lattice vectors. None is the
        local response, the average over the q of the k-mesh. A finite (0D)
        system has no momentum and gives the same response for any q.
    energies: frequencies.
    U: residual interaction U'(q) of the paper, dressed in the random phase
        approximation as the exchange 2U between the spin components of
        the moments, which is 1/(1/chi - U) for one moment per cell with
        spin rotation symmetry. It is meaningful while 1 - U chi(q,0) > 0;
        at zero the mean field is unstable to magnetic order at q.
        A number, a matrix between the moments of the unit cell, or a
        function of q (the three reduced coordinates) that returns one of
        them. None, the default, is the bare mean-field response.
    nk: k-mesh of the bubble for each periodic direction.
    delta: broadening of the frequency.
    T: temperature of the occupations, equal to delta when not given (the
        convention of the spin responses of this library).
    hybridization_scale: factor that multiplies the hybridization V of the
        mean field before the response is computed, 1.5 in the comparison
        of the single-particle bands of the paper.

    Returns (energies, chi), with chi of shape (number of energies, number
    of moments, number of moments), in the sign convention of the spin
    responses of this library: the static response is negative and the
    spectral weight at positive frequency is +Im chi/pi."""
    what = "the spin susceptibility of the local moments"
    pairs = pairstk.get_pairs(h, what)
    energies = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    if delta <= 0.0:
        raise ValueError("delta must be positive, the broadening of the "
                "frequency; got "+str(delta))
    h = pairstk.scale_hybridization(h, hybridization_scale, pairs)
    fsites = [fi for (ci, fi) in pairs]
    if U is None: # the bare response needs the matrix elements of S^+ alone
        return localspinchi.bare_raising_chi(h, q=q, energies=energies,
                nk=nk, delta=delta, T=T, sites=fsites)
    if q is None: # the local response, every q dressed on its own
        out, nq = 0., 0
        for qi, chi in localspinchi.bare_spin_chi_on_mesh(h,
                energies=energies, nk=nk, delta=delta, T=T, sites=fsites):
            out = out + _moment_response(chi, qi, U, len(fsites))
            nq += 1
        return energies, out/nq
    chi = localspinchi.bare_spin_chi(h, q=q, energies=energies, nk=nk,
            delta=delta, T=T, sites=fsites)[1] # (energies, 3nf, 3nf)
    return energies, _moment_response(chi, q, U, len(fsites))


def _moment_response(chi, q, U, nf):
    """The transverse response of the moments at one q from the bare
    response chi of their spin operators dressed with the residual
    interaction U, see moment_spinchi"""
    # U S_q.S_-q is the exchange 2U between the spin components
    W = np.kron(np.identity(3), 2.*_residual_at_q(U, q, nf))
    chi = localspinchi._dyson(chi, W)
    x, y = slice(0, nf), slice(nf, 2*nf)
    # S^+ = Sx + i Sy on the first index, S^- = Sx - i Sy on the second:
    # in the pair sum of this library the operator of the first index takes
    # the occupied state to the empty one, so S^+ is the one that excites
    return chi[:, x, x] + chi[:, y, y] - 1j*chi[:, x, y] + 1j*chi[:, y, x]
