"""SpinonHamiltonian: Abrikosov-pseudofermion (auxiliary-fermion) mean-field
theory for spin-1/2 Heisenberg models.

Represents each spin as S_i = 1/2 f_i^dagger sigma f_i (Abrikosov fermions),
subject to the hard local constraint f_i^dagger f_i = 1 (exactly one fermion
per site), and self-consistently decouples J S_i.S_j into an RVB bond order
parameter chi_ij = <f_i^dagger f_j> -- see Savary & Balents, "Quantum Spin
Liquids: a review", arXiv:1601.03742, Sec. 4 ("Partons") and Sec. 4.1 in
particular, whose Eq. 35/36/37/38 and surrounding text this class follows
directly. Physically this is exactly meanfield.VJinteraction's J-channel
(already the correct Fock/Hartree-Fock decoupling of a spin-spin exchange,
see scftk/spinspin.py's module docstring) run on a Hamiltonian
with zero bare hopping (a pure spin model has no bare electron tunneling --
every "kinetic" term is the auxiliary mean field itself) and with the local
constraint enforced exactly, at every site, via the per-site `filling`
array VJinteraction now supports (scftk.spinspin._run_anisotropic_
scf's array-filling branch) instead of only a single lattice-averaged Fermi
level.

Only the U(1) (RVB bond-only) ansatz is implemented -- a Z2 ansatz (also
allowing the pairing/anomalous channel J already induces on a BdG
Hamiltonian, see meanfield.VJinteraction's docstring) would need a
Nambu-doubled SpinonHamiltonian and a has_eh=True-aware local constraint,
neither of which exist yet (VJinteraction's array-filling path explicitly
raises NotImplementedError for has_eh=True -- see
scftk.spinspin._run_anisotropic_scf).

Two decouplings of the exchange are available, decoupling= of
get_mean_field_hamiltonian:

"hartree-fock" (the default) is the unrestricted Hartree-Fock one
described above. Its mean field on a bond is 3J/8 times chi_ij for a
spin-singlet state, and it has the magnetic channel too, so on most
lattices the loop leaves the spin liquid for a magnetically ordered state
whenever the cell allows one.

"bond" keeps the bond channel alone, with the full weight of the identity
S_i.S_j = -1/2 chihat^dag_ij chihat_ij + const (chihat_ij = sum_s
f^dag_is f_js) that holds at one fermion per site, so the hopping of the
spinons is t_ij = J_ij chi_ij / 2 and there is no magnetic mean field.
This is the saddle point of Willsher and Knolle, arXiv:2503.13831
(Methods, "Parton mean field theory"), which they pair with a random
phase approximation in the spin channel, chi = chi0 (1 + J(q) chi0)^-1
in the usual sign convention: get_spinchi_full does that for a
Hamiltonian converged this way, and get_rpa_stiffness gives the smallest
eigenvalue of the static kernel, which is negative when the spin liquid
is unstable towards magnetic order at that momentum. The loop is the
Hartree-Fock one restricted to its spin-singlet part with the exchange
multiplied by 4/3, which is the same mean field (3/8 times 4/3 is 1/2).
Checked against that paper on the pi-flux state of the triangular
lattice: t/J = 0.198, a static bubble of 0.389/J at K and 0.425/J at M,
and the instability to 120 degree order at J2/J1 = 0.0716
(tests/spinon/test_spinon_bond_rpa.py).

gutzwiller= multiplies the exchange in the mean field and in the vertex,
the approximate account of the Gutzwiller projection of that paper, which
uses 2 (the local moment S^2 is 3/8 in the unprojected state and 3/4 in
the projected one). It rescales the energies and leaves the stability
unchanged."""
import numpy as np

from . import algebra
from .hamiltonians import Hamiltonian, _mean_field_scf_result
from .meanfield import VJinteraction

# factor of the exchange inside the Hartree-Fock loop for each decoupling:
# the spin-singlet part of the Hartree-Fock mean field of J S_i.S_j is
# 3J/8 chi_ij, and the bond decoupling wants J/2 chi_ij
_DECOUPLINGS = {"hartree-fock": 1.0, "bond": 4./3.}

# keyword arguments of the spin response that chitk.localspinchi takes
_LOCAL_CHI_KWARGS = {"q", "energies", "nk", "delta", "T"}


def get_decoupling_names():
    """Names accepted by decoupling= of
    SpinonHamiltonian.get_mean_field_hamiltonian"""
    return list(_DECOUPLINGS)


def _is_zero(x):
    """True for an interaction left at its default: None or the number 0"""
    return x is None or (np.isscalar(x) and x == 0)


def _ansatz_phases(ansatz, n, tol=1e-12):
    """The phase of every hopping of an ansatz, {(n1,n2,n3): N x N matrix
    of unit modulus, zero where the ansatz has no hopping}, read from its
    spin-up block. ansatz is a Hamiltonian, a dictionary of hopping
    matrices or the matrix of the cell, as a mean-field guess, for a cell
    of n sites."""
    from .scftk.mfconstrains import obj2mf
    out = {}
    for d, m in obj2mf(ansatz).items():
        m = np.asarray(algebra.todense(m))
        if m.shape != (2*n, 2*n):
            raise ValueError("ansatz must be spinful, with hopping matrices "
                    "of size %d x %d for the %d sites of the cell, got %s; "
                    "build it with has_spin=True"
                    % (2*n, 2*n, n, " x ".join(str(i) for i in m.shape)))
        m = m[0::2, 0::2]
        a = np.abs(m)
        out[tuple(int(i) for i in d)] = np.where(a > tol,
                m/np.where(a > tol, a, 1.), 0.)
    return out


def _bond_projection(phases=None):
    """Return the function that restricts a mean field to the bond
    channel: its spin-singlet part, the same for the two spins and with no
    element between them. With phases (see _ansatz_phases) every hopping
    also keeps the phase it has in the ansatz, with a real amplitude, and
    a hopping absent from the ansatz stays zero; the onsite energies are
    not touched by that."""
    def f(mf):
        out = {}
        for d, m in mf.items():
            m = np.asarray(algebra.todense(m))
            s = (m[0::2, 0::2] + m[1::2, 1::2])/2. # spin-singlet part
            if phases is not None:
                key = tuple(int(i) for i in d)
                p = phases.get(key, None)
                onsite = np.diag(s).copy()
                if p is None: s = np.zeros_like(s)
                else: s = p*np.real(np.conj(p)*s)
                if key == (0, 0, 0): np.fill_diagonal(s, onsite)
            new = np.zeros(m.shape, dtype=np.complex128)
            new[0::2, 0::2] = s
            new[1::2, 1::2] = s
            out[d] = new
        return out
    return f


class SpinonHamiltonian(Hamiltonian):
    """A Hamiltonian for Abrikosov-pseudofermion mean-field theory of a
    spin-1/2 Heisenberg model on `geometry`.

    Built with zero bare hopping (see the module docstring) -- exchange
    couplings are supplied entirely through get_mean_field_hamiltonian's
    J1/J2/J3/Jr/J1x/J1y/J1z kwargs (meanfield.VJinteraction's own
    convention), e.g.::

        from pyqula import geometry
        from pyqula.spinon import SpinonHamiltonian
        g = geometry.triangular_lattice()
        h = SpinonHamiltonian(g)
        h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=12)
        h2.local_occupation   # <n_i> per site, target is exactly 1.0
        h2.constraint_lambda  # converged per-site Lagrange multipliers

    `filling` cannot be overridden -- the auxiliary-fermion representation
    is only valid at exactly one fermion per site, everywhere, so this
    class always requests that (as the per-site array VJinteraction's
    array-filling path expects, one entry per site, in its 0-to-1
    fraction-of-2-orbital-capacity convention -- see
    densitymatrix.full_dm_accumulate_sparse_local_fermi's docstring), never
    a value the caller supplies. Any interaction/SCF kwarg
    get_mean_field_hamiltonian otherwise accepts (V1/V2/V3/U for an
    additional density-density term, mf=, nk=, mix=, maxerror=, maxite=,
    constrains=, ...) is forwarded unchanged; `scf.converged` (behind
    get_mean_field_hamiltonian's usual MeanFieldConvergenceError on
    non-convergence, see hamiltonians._mean_field_scf_result) already
    implies the local constraint itself converged to within `maxerror`, since
    _run_anisotropic_scf folds the per-site occupation residual into the
    same convergence check as the mean field's own -- no separate
    tolerance to track.

    An external Zeeman/magnetic field couples EXACTLY (not via any
    mean-field decoupling) to the auxiliary fermion, since
    S_i = 1/2 f_i^dagger sigma f_i is itself already bilinear in f: physically
    it is just an extra single-particle term -h.S_i, i.e. -(h/2).sigma_i,
    added to the auxiliary Hamiltonian before the RVB exchange mean field is
    layered on top. Use the inherited Hamiltonian.add_zeeman/add_exchange
    (called on this instance BEFORE get_mean_field_hamiltonian, same as for
    any other Hamiltonian in this codebase) to add one::

        h = SpinonHamiltonian(g)
        h.add_zeeman([0., 0., 0.3])  # or h.add_exchange([0.,0.,0.3])
        h2 = h.get_mean_field_hamiltonian(J1=1.0, nk=12)

    Note add_zeeman/add_exchange's argument b is the coefficient of sigma
    (Pauli matrices, eigenvalues +-1), not of S=sigma/2, and it enters as
    +b.sigma = +2b.S_i -- so the physical field h in H=-h.S_i is -2*b, and
    the induced <S_i> points AGAINST b, matching this argument's convention
    everywhere else in pyqula (add_exchange/add_magnetism on an ordinary
    electronic Hamiltonian use the same sigma-not-S convention). The local
    constraint enforced by the array-filling machinery above (exactly one
    fermion per site, summed over spin) is unaffected by the field -- it is
    a total-occupation constraint, not a constraint on the spin
    polarization, so a finite field is free to induce a net <S_i> (up to
    full polarization at large field/J) while local_occupation stays
    exactly 1.0 (verified in tests/spinon/test_spinon_zeeman.py: an
    isotropic response to same-magnitude fields along different axes,
    saturation at large field, and constraint preservation throughout --
    see Savary & Balents, arXiv:1601.03742, Sec. 4.1, on the field response
    of a U(1) RVB spin liquid)."""

    def __init__(self, geometry=None):
        super().__init__(geometry)
        if geometry is not None:
            h0 = geometry.get_hamiltonian(tij=[0.0], has_spin=True)
            self.set_multihopping(h0.get_multihopping())
            self.is_multicell = h0.is_multicell

    def get_mean_field_hamiltonian(self, return_total_energy=False,
            decoupling="hartree-fock", gutzwiller=1.0, ansatz=None,
            **kwargs):
        """Self-consistent spinon mean field of the exchange J1/J2/J3/Jr,
        at one fermion per site. See the class docstring for the keyword
        arguments shared with meanfield.VJinteraction.

        decoupling selects the mean-field decoupling of the exchange, one
        of spinon.get_decoupling_names(): "hartree-fock" (the default) or
        "bond", the bond channel alone with t_ij = J_ij chi_ij / 2 and no
        magnetic channel (see the module docstring). The "bond" one takes
        an isotropic exchange only, no density-density interaction, no
        Zeeman field, and integration="ed". Its total energy
        (return_total_energy=True) is the saddle-point value of the bond
        functional, -J_ij |chi_ij|^2 / 2 per bond (times gutzwiller), not
        the expectation value of the Heisenberg model in the state, which
        is 3/4 of it at gutzwiller=1.

        gutzwiller multiplies every exchange coupling, in the mean field
        and in the vertex of the spin response.

        ansatz, for decoupling="bond", is a hopping pattern (a
        Hamiltonian, a dictionary of hopping matrices or the matrix of the
        cell, as mf=) that the loop holds: every hopping keeps the phase
        it has in the ansatz and converges a real amplitude, and a hopping
        that is zero in the ansatz stays zero. It is also the initial
        guess when mf is not given. Without it the loop is free to leave a
        named state for another one of lower mean-field energy, e.g. the
        pi-flux state of the triangular lattice for the chiral one.

        A Hamiltonian converged with decoupling="bond" carries the
        exchange it was converged with in h.Vchannels and
        h.site_spin_vertex=True, which is what makes get_spinchi_full,
        get_spinchi_ladder and get_magnon_bands dress the response with
        the site-basis vertex J(q) rather than with the pair-basis ladder
        of the Hartree-Fock mean field."""
        if "filling" in kwargs:
            raise ValueError("SpinonHamiltonian enforces exactly one "
                    "Abrikosov fermion per site -- filling= is fixed by "
                    "the representation itself and cannot be overridden "
                    "(see the class docstring)")
        if decoupling not in _DECOUPLINGS:
            raise ValueError("unknown decoupling %r of the spinon mean "
                    "field; the accepted ones are %s" % (decoupling,
                    ", ".join(repr(d) for d in get_decoupling_names())))
        if not (np.isscalar(gutzwiller) and gutzwiller > 0):
            raise ValueError("gutzwiller must be a positive number, the "
                    "factor of the exchange, got %r" % (gutzwiller,))
        bond = decoupling == "bond"
        if ansatz is not None and not bond:
            raise NotImplementedError("ansatz= holds the hopping pattern "
                    "of decoupling='bond' only, got decoupling=%r; pass "
                    "the pattern as the initial guess mf= instead"
                    % (decoupling,))
        if bond: self._require_bond_decoupling(kwargs)
        kw = dict(kwargs)
        scale = gutzwiller*_DECOUPLINGS[decoupling]
        for name in ("J1", "J2", "J3", "J1x", "J1y", "J1z"):
            if name in kw: kw[name] = scale*kw[name]
        Jr = kw.get("Jr", None)
        if Jr is not None:
            kw["Jr"] = lambda *args, **kws: scale*Jr(*args, **kws)
        if bond:
            phases = None if ansatz is None else _ansatz_phases(ansatz,
                    len(self.geometry.r))
            from .scftk.spinspin import _compose_callbacks
            kw["callback_mf"] = _compose_callbacks(
                    kw.get("callback_mf", None), _bond_projection(phases))
            if ansatz is not None and kw.get("mf", None) is None:
                from .scftk.mfconstrains import obj2mf
                kw["mf"] = {d: np.array(algebra.todense(m),
                        dtype=np.complex128)
                        for d, m in obj2mf(ansatz).items()}
        n = len(self.geometry.r)
        scf = VJinteraction(self, filling=np.full(n, 0.5), **kw)
        if scf.converged:
            # local_occupation in the <n_i> in [0,2] electron-count
            # convention (matching f_i^dagger f_i directly -- the target is
            # exactly 1.0), not VJinteraction's internal 0-to-1 fraction
            # one scf.local_occupation itself uses -- see
            # densitymatrix.full_dm_accumulate_sparse_local_fermi's
            # docstring for that distinction.
            scf.hamiltonian.local_occupation = 2.0*scf.local_occupation
            scf.hamiltonian.constraint_lambda = np.asarray(scf.lam)
            scf.hamiltonian.gutzwiller = gutzwiller
            if bond:
                self._record_bond_exchange(scf.hamiltonian, gutzwiller,
                                           kwargs)
        return _mean_field_scf_result(scf, return_total_energy)

    def _require_bond_decoupling(self, kwargs):
        """Raise for what decoupling="bond" does not take"""
        from .chitk.localspinchi import is_spin_degenerate
        for name in ("J1x", "J1y", "J1z"):
            if not _is_zero(kwargs.get(name, 0.0)):
                raise NotImplementedError("decoupling='bond' is implemented "
                        "for an isotropic exchange (J1, J2, J3, Jr) only, "
                        "got %s=%r; use decoupling='hartree-fock' for an "
                        "anisotropic one" % (name, kwargs[name]))
        for name in ("V1", "V2", "V3", "U", "Vr"):
            if not _is_zero(kwargs.get(name, 0.0)):
                raise NotImplementedError("decoupling='bond' is implemented "
                        "for the exchange alone, got the density-density "
                        "interaction %s=%r; use decoupling='hartree-fock' "
                        "to combine the two" % (name, kwargs[name]))
        integration = kwargs.get("integration", "ed")
        if integration != "ed":
            raise NotImplementedError("decoupling='bond' is implemented "
                    "with integration='ed' only, got integration=%r"
                    % (integration,))
        if not is_spin_degenerate(self):
            raise NotImplementedError("decoupling='bond' is not implemented "
                    "for a Hamiltonian that depends on the spin (a Zeeman "
                    "field or an exchange field): it has no magnetic "
                    "channel, so the field of the neighboring spins would "
                    "be missing; use decoupling='hartree-fock'")

    @staticmethod
    def _record_bond_exchange(h, gutzwiller, kwargs):
        """Store on the converged Hamiltonian the exchange it was
        converged with, as the three spin channels the site-basis spin
        vertex is built from (chitk.spinchi._channel_spin_U), replacing
        the ones the loop recorded, which carry the factor 4/3 of the
        bond decoupling."""
        from .scftk.spinspin import _build_v
        Jr = kwargs.get("Jr", None)
        Jrg = None if Jr is None else \
                (lambda *args, **kws: gutzwiller*Jr(*args, **kws))
        v = _build_v(h, gutzwiller*kwargs.get("J1", 0.0),
                gutzwiller*kwargs.get("J2", 0.0),
                gutzwiller*kwargs.get("J3", 0.0), Jrg,
                rcut=kwargs.get("rcut", None))
        h.V = v
        h.Vchannels = {"x": v, "y": v, "z": v, "d": None}
        h.site_spin_vertex = True

    def _refuse_pair_ladder(self, what):
        if getattr(self, "site_spin_vertex", False):
            raise NotImplementedError("%s sums the ladder of the "
                    "Hartree-Fock mean field in the pair basis, which is "
                    "not the kernel of a spinon mean field converged with "
                    "decoupling='bond'; use get_spinchi_full, "
                    "get_spinchi_ladder or get_magnon_bands(method='rpa'), "
                    "which dress it with the site-basis vertex J(q)"
                    % (what,))

    def get_spinchi_full(self, RPA=True, **kwargs):
        """Full spin response, (energies, one 3N x 3N matrix per energy).

        The same response as Hamiltonian.get_spinchi_full, with the bubble
        of chitk.localspinchi when only q, energies, nk, delta and T are
        given and the CPU is the backend: it builds the matrix elements of
        the local spin operators directly and is the one that reaches unit
        cells of a few hundred sites. For a mean field converged with
        decoupling="bond", RPA=True dresses the bubble with the exchange
        in the spin channel, chi0 (1 + J(q) chi0)^-1 in the usual sign
        convention (chi0 (1 - J(q) chi0)^-1 in the one of this library,
        whose static response is negative), arXiv:2503.13831. For the
        Hartree-Fock one RPA=True is the pair-basis ladder, as for any
        Hamiltonian."""
        from . import gpu
        from .chitk import localspinchi, spinchi
        local = set(kwargs) <= _LOCAL_CHI_KWARGS and not gpu.get_gpu()
        if not local or (RPA and spinchi._use_pair_basis(self)):
            return super().get_spinchi_full(RPA=RPA, **kwargs)
        V = spinchi._full_spin_U(self) if RPA else None
        return localspinchi.spin_chi(self, V=V, **kwargs)

    def get_rpa_stiffness(self, q, **kwargs):
        """Return the smallest eigenvalue of the static kernel
        1 + J(q) chi0(q,0) of the spin-channel RPA of a mean field
        converged with decoupling="bond".

        It is 1 without exchange, it decreases as the spin liquid
        approaches a magnetic instability at momentum q (in units of the
        reciprocal lattice vectors of the cell), and a negative value
        means that the liquid is unstable there. This is the stiffness of
        arXiv:2503.13831 up to the factor -J(q). nk, delta and T are those
        of the bubble (chitk.localspinchi.static_kernel_eigenvalues)."""
        if not getattr(self, "site_spin_vertex", False):
            raise ValueError("get_rpa_stiffness needs a spinon mean field "
                    "converged with decoupling='bond', whose spin response "
                    "is dressed in the spin channel; this Hamiltonian is "
                    "not one")
        from .chitk import localspinchi, spinchi
        ev = localspinchi.static_kernel_eigenvalues(self,
                spinchi._full_spin_U(self), q, **kwargs)
        return ev[0].real

    def get_transverse_spinchi(self, **kwargs):
        self._refuse_pair_ladder("get_transverse_spinchi")
        return super().get_transverse_spinchi(**kwargs)

    def get_magnon_bands(self, method="rpa", **kwargs):
        if method in ("pair", "tdhf"):
            self._refuse_pair_ladder("get_magnon_bands(method=%r)" % method)
        return super().get_magnon_bands(method=method, **kwargs)

    def get_magnon_energies(self, **kwargs):
        self._refuse_pair_ladder("get_magnon_energies")
        return super().get_magnon_energies(**kwargs)

    def get_goldstone_residual(self, **kwargs):
        self._refuse_pair_ladder("get_goldstone_residual")
        return super().get_goldstone_residual(**kwargs)
