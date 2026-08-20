"""Static RPA screened interaction on top of a mean-field Hamiltonian.

Public entry point composing the bsetk/screening.py internals, in the same
way chi.py composes chitk/ and bse.py composes bsetk/.

The physics: a mean-field calculation is converged with some interaction
v, but v is not what an added electron and hole actually feel between
them. The rest of the electrons rearrange around them, and the interaction
that survives that rearrangement is the screened one,

    W(q) = eps^-1(q) v(q),   eps(q) = 1 - v(q) chi0(q)

with chi0 the static polarizability of the very bands the mean field just
produced. Because the mean-field step already leaves {e_n(k), C^n(k)} on a
k-mesh, computing W costs little more than the Brillouin-zone sum for
chi0, and it is what turns a time-dependent-Hartree-Fock BSE into a
GW-BSE-style one (h.get_bse(screening="rpa")).

W is also useful on its own: .get_dict() returns it as a real-space
interaction, which can be inspected to see how far the screened
interaction actually reaches, or fed back into
get_mean_field_hamiltonian(V=...) for a screened-exchange mean field.

READ THIS BEFORE USING IT. A Hubbard U fitted to reproduce a material is
already an effective, screened interaction; screening it again is double
counting and gives a spuriously weak interaction. This machinery is for a
genuinely BARE interaction -- a long-range Coulomb tail built with
bsetk.interaction.density_interaction(Vr=...), or bare model V1/V2/V3
shells. See bsetk/screening.py's module docstring for the rest of the
conventions and traps.
"""

from .bsetk.screening import (ScreenedInteraction, screened_interaction,
        static_polarizability)


def get_screened_interaction(h,**kwargs):
    """Return the static RPA screened interaction of a Hamiltonian, as a
    ScreenedInteraction (see screened_interaction)"""
    return screened_interaction(h,**kwargs)


def get_polarizability(h,**kwargs):
    """Return (qs,chi0), the static polarizability of a Hamiltonian on its
    k-mesh (see static_polarizability)"""
    return static_polarizability(h,**kwargs)
