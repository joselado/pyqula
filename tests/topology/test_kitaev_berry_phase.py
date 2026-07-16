import numpy as np

from pyqula import geometry
from pyqula.topology import berry_phase

T = 1.0  # chain hopping amplitude


def _phase(mu):
    """Berry phase (in units of pi) of a spin-polarized chain with an
    in-plane p-wave pairing, isolated by a large Zeeman field, as in
    examples/1d/kitaev/main.py. `mu` is the chemical potential of the
    isolated (spin-down) band."""
    g = geometry.chain()
    h = g.get_hamiltonian()
    h.add_onsite(20 + mu)
    h.add_zeeman([0., 0., 20.])
    h.add_pairing(mode="pwave", delta=0.3, d=[1., 0., 0.])
    return berry_phase(h, nk=100, write=False) / np.pi


def test_kitaev_chain_berry_phase_is_quantized_and_flips_at_transition():
    """The Berry (Zak) phase of the 1D Kitaev chain is quantized to 0 or pi,
    and switches between the two at the analytically known topological
    transition |mu| = 2*t."""
    for mu in (-3.0, -1.0, 0.0, 1.0, 3.0):
        phi = _phase(mu)
        assert min(abs(phi), abs(phi - 1), abs(phi + 1)) < 1e-6, \
            f"Berry phase not quantized at mu={mu}: {phi}"

    trivial = round(_phase(2.5 * T)) % 2
    topological = round(_phase(0.0)) % 2
    assert trivial != topological
