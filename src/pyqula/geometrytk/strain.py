

def add_strain(g,s):
    """Add strain to a geometry"""
    if g.dimensionality!=2: raise # only for 2D
    g.a1 *= 1.+s 
    g.a2 *= 1-s 
    g.fractional2real()

