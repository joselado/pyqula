"""GrapheneGeometry: a Geometry that knows how to relax itself.

Wraps any graphene multilayer geometry (bilayer, twisted bilayer, twisted
trilayer, ...) built the usual way (e.g. via specialgeometry.twisted_bilayer
or by stacking honeycomb_lattice() copies) and adds a .relax() method that
runs the phenomenological GSFE+elastic relaxation of relax.py. See that
module's docstring, and gsfe.py/elastic.py, for the underlying physics
(arXiv:1805.06972)."""
from ..geometry import Geometry
from .relax import relax_structure


class GrapheneGeometry(Geometry):
    """A graphene multilayer Geometry with a .relax() method.

    Example::

        from pyqula import specialgeometry
        from pyqula.graphenetk.geometry import GrapheneGeometry
        g = GrapheneGeometry(specialgeometry.twisted_bilayer(m0=15))
        g = g.relax()  # AA area shrinks, AB/BA domains grow
    """

    def __init__(self, g=None):
        super().__init__()
        if g is None:
            return
        if not getattr(g, "has_sublattice", False):
            raise ValueError("GrapheneGeometry needs a geometry with "
                              "has_sublattice=True (e.g. built from "
                              "geometry.honeycomb_lattice())")
        self.__dict__.update(g.__dict__)

    def relax(self, **kwargs):
        """Return a new, relaxed GrapheneGeometry minimizing the GSFE
        interlayer + elastic intralayer energy over an in-plane
        displacement field. See relax.relax_structure for kwargs
        (nrep, maxiter, verbose, layer_pairs, gsfe_coeffs,
        elastic_coeffs)."""
        return relax_structure(self, **kwargs)
