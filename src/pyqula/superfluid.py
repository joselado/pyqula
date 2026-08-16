"""Superfluid weight (superfluid stiffness) of Bogoliubov-de Gennes
mean-field Hamiltonians, and the associated BKT temperature.

This is the public entry point; the formalism, conventions, units,
validity limits and references live in sctk/superfluidweight.py, which
does the actual work.  In short:

    D_s^{ab} = (1/V) d^2 Omega / dQ_a dQ_b

with the pairing amplitude |Delta| frozen and its phase wound (Q is a
Cartesian twist wavevector), evaluated either from the analytic multiband
Kubo formula of Liang et al., PRB 95, 024515 (2017) (the default, and the
ground truth here) or from a direct finite difference of the grand
potential.  The quantum-metric integral is *not* used as a definition: the
conventional/geometric split is offered as a decomposition of the Kubo
result, and is refused when its assumptions (uniform on-site pairing,
time-reversal symmetry) do not hold.

The twist is the physical Peierls substitution, with the *full* bond
vector R + r_j - r_i, not just the lattice vector R -- see the
sctk/superfluidweight.py docstring, and the gauge argument below.

Typical use::

    from pyqula import geometry
    g = geometry.square_lattice()
    h = g.get_hamiltonian()
    h.add_onsite(-0.6)          # move away from half filling
    h.add_swave(0.3)            # BdG Hamiltonian with an s-wave gap
    D = h.get_superfluid_weight(nk=20)          # 2x2 Cartesian tensor
    out = h.get_superfluid_weight(nk=20,decompose=True)
    print(out["conventional"],out["geometric"])
    print(h.get_bkt_temperature(nk=20))
"""

from .sctk import superfluidweight


def superfluid_weight(h,mode="kubo",decompose=False,**kwargs):
    """Superfluid weight tensor D_s^{ab} of a BdG (Nambu) Hamiltonian.

    Parameters
    ----------
    h : Hamiltonian
        Must be in a Nambu mode (h.add_swave(...) / h.turn_nambu()), with
        dimensionality 1, 2 or 3.
    mode : {"kubo","finite_difference"}
        "kubo" (default) uses the analytic multiband mean-field formula.
        "finite_difference" differentiates the grand potential with respect
        to the twist numerically -- much slower, but assumption free; it is
        the oracle the analytic route is tested against.
    decompose : bool
        If True, return a dictionary with the total weight and its
        conventional and quantum-geometric parts instead of a bare tensor.
        Only available under uniform on-site pairing and time-reversal
        symmetry; a ValueError is raised otherwise rather than reporting a
        meaningless split.  Ignored for mode="finite_difference".
    nk : int
        Linear size of the BZ mesh (default 20).
    T : float
        Temperature, in the units of the Hamiltonian (default 0, exact).

        Beware the T=0 limit when the *normal* state is gapless and the
        pairing is zero or very small: there the paramagnetic and
        diamagnetic terms are supposed to cancel, but the cancellation is
        carried by a -df/dE that collapses to a delta function at T=0, so a
        finite k-mesh cannot resolve it and D_s comes out at the normal
        state's Drude weight instead of zero (e.g. a honeycomb lattice at
        zero pairing gives D_xx = 0.1 at T=0, but 1e-7 at T=0.2/nk=24 and
        1e-16 at T=0.4/nk=40).  This is physics, not a bug -- the same
        expression *is* the Drude weight at T=0 -- but if you are checking
        that a marginal state has no stiffness, use a temperature the mesh
        resolves.  It does not affect a properly gapped superconductor,
        where the Bogoliubov spectrum has no zero-energy states.
    gauge : {"atomic","lattice"}
        "atomic" (default) twists with the full bond vector R + r_j - r_i,
        which is the physical Peierls substitution.  "lattice" drops the
        intracell part and twists with R alone, reproducing the cell-gauge
        convention of the Peotta/Toermae papers and of pyqula's own
        h.get_quantum_metric(); the two differ only for cells holding more
        than one orbital.

    Returns
    -------
    ndarray of shape (dim,dim) in Cartesian twist coordinates, or a
    dictionary if decompose is True.
    """
    if mode in ["kubo","analytic"]:
        if decompose:
            return superfluidweight.superfluid_weight_decomposition(h,**kwargs)
        return superfluidweight.superfluid_weight(h,**kwargs)
    elif mode in ["finite_difference","fd"]:
        return superfluidweight.superfluid_weight_finite_difference(h,**kwargs)
    else: raise ValueError("unknown superfluid weight mode "+str(mode))


def bkt_temperature(h,**kwargs):
    """Berezinskii-Kosterlitz-Thouless temperature of a 2d BdG Hamiltonian
    from the Nelson-Kosterlitz criterion T_BKT = (pi/8) D_s(T_BKT), solved
    self-consistently by bisection at fixed |Delta|.  The temperature is
    the unknown being solved for, so it cannot be passed in.  See
    sctk/superfluidweight.bkt_temperature."""
    if "T" in kwargs: raise TypeError("bkt_temperature solves for the "
            "temperature, so T cannot be given")
    return superfluidweight.bkt_temperature(h,**kwargs)


def grand_potential(h,**kwargs):
    """Twist-dependent part of the BdG grand potential per unit cell.  Only
    differences in the (Cartesian) twist Q are meaningful, since a
    Q-independent constant is dropped; this is the quantity the
    finite-difference route differentiates."""
    return superfluidweight.grand_potential(h,**kwargs)
