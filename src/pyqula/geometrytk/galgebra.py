import numpy as np

from ..geometry import Geometry

def sum_geometries(g1,g2):
    """Sum two geometries"""
    if type(g2)==Geometry:
        if g1.dimensionality!=g2.dimensionality: raise
        g = g1.copy()
        g.r = np.concatenate([g1.r,g2.r])
        g.r2xyz()
        if g.has_sublattice:
            g.sublattice = np.concatenate([g1.sublattice,g2.sublattice])
        if g.atoms_have_names:
            g.atoms_names = np.concatenate([g1.atoms_names,g2.atoms_names])
        if g1.primal_geometry is not None and g2.primal_geometry is not None:
            print("Primal geometry not implemented in __add__")
            raise
        return g
    elif type(g2)==np.ndarray: # array input
        g = g1.copy() # copy geometry
        g.r = g.r + g2 # shift all the positions
        g.r2xyz()
        return g
    else:
        print(type(g2))
        raise

