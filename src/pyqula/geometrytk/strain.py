

def add_strain(g,s):
    """Add strain to a geometry"""
    if g.dimensionality!=2: # only for 2D
        raise ValueError("strain of a geometry is only implemented in 2d")
    g.a1 *= 1.+s 
    g.a2 *= 1-s 
    g.fractional2real()

