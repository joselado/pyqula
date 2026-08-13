"""GrapheneHamiltonian: a Hamiltonian built from a graphene multilayer
geometry (typically a, possibly relaxed, GrapheneGeometry), defaulting to
the distance-decaying hoppings of specialhopping.twisted_matrix instead of
the generic first-neighbor default -- these hoppings depend on the true
3D interatomic distance, so a relaxed (in-plane displaced) geometry feeds
into the electronic structure automatically, with no change needed here."""
from ..hamiltonians import Hamiltonian
from .. import specialhopping


class GrapheneHamiltonian(Hamiltonian):
    """Example::

        from pyqula import specialgeometry
        from pyqula.graphenetk.geometry import GrapheneGeometry
        from pyqula.graphenetk.hamiltonian import GrapheneHamiltonian
        g = GrapheneGeometry(specialgeometry.twisted_bilayer(m0=15)).relax()
        h = GrapheneHamiltonian(g)
        (k,e) = h.get_bands()
    """

    def __init__(self, geometry=None, ti=0.12, lambi=8.0, lamb=12.0, dl=3.0,
                 has_spin=False, is_sparse=True, mgenerator=None, **kwargs):
        super().__init__(None)
        if geometry is None:
            return
        if mgenerator is None:
            mgenerator = specialhopping.twisted_matrix(
                ti=ti, lambi=lambi, lamb=lamb, dl=dl)
        h0 = geometry.get_hamiltonian(has_spin=has_spin, is_sparse=is_sparse,
                                       is_multicell=True,
                                       mgenerator=mgenerator, **kwargs)
        super().__init__(h0.geometry)
        self.set_multihopping(h0.get_multihopping())
        self.is_multicell = h0.is_multicell
