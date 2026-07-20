"""WannierHamiltonian: the Hamiltonian subclass returned by
``wannierize.get_wannier_hamiltonian``.
"""
from ..hamiltonians import Hamiltonian


class WannierHamiltonian(Hamiltonian):
    """An ordinary pyqula ``Hamiltonian`` -- every method (``get_bands``,
    ``get_dos``, ``get_chern``, ...) is inherited unchanged, since a
    Wannierized Hamiltonian's real-space hoppings already carry the full
    physics of the Wannierized band subspace and need no special-casing.

    This subclass only exists to also carry Wannierization-specific data
    alongside the Hamiltonian itself:

    wannier_functions : dict
        ``{R: (num_orbitals, num_wann) complex ndarray}``, the real-space
        Wannier function coefficients expressed in the *original*
        Hamiltonian's orbital basis (same bare, tau-free periodic-gauge
        convention as ``get_hk_gen()``, see ``htk/bloch.py``): column n,
        row o is the amplitude of Wannier function n -- translated to
        cell R relative to the home cell -- on orbital o of the original
        Hamiltonian. See ``wannierize._wannier_functions_from_gauge`` for
        how these are reconstructed from the wannier90 CG gauge matrix.
    wannier_band_indices, wannier_clusters, wannier_centres,
    wannier_spreads, wannier_spread_total, wannier_setup_result,
    wannier_run_result, wannier_particle_hole_operator :
        Wannierization diagnostics -- see
        ``wannierize.get_wannier_hamiltonian``'s docstring.
    """
    def __init__(self, geometry=None):
        super().__init__(geometry)
        self.wannier_functions = {}
        self.wannier_band_indices = None
        self.wannier_clusters = None
        self.wannier_centres = None
        self.wannier_spreads = None
        self.wannier_spread_total = None
        self.wannier_setup_result = None
        self.wannier_run_result = None
        self.wannier_particle_hole_operator = None
