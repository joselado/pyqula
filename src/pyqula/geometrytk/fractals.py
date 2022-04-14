# library for fractal geometries
import numpy as np

def sierpinski(n=4,mode="triangle"):
    """Create a sierpinski geometry"""
    from .. import supercell
    from ..geometry import triangular_lattice
    from .. import geometry
    g = triangular_lattice() # get geometry
    if mode=="triangular":
        r = np.array([[0.,0.,0.]])
        a1 = np.array([1.,0.,0.])
        a2 = np.array([np.cos(np.pi/3),np.sin(np.pi/3),0.])
        vs = [a1,a2]
        inflate = lambda i: 2**i # inflate function
    elif mode=="square":
        r = np.array([[0.,0.,0.]])
        a1 = np.array([1.,0.,0.])
        a2 = np.array([0.,1.,0.])
        vs = [a1,2*a1,a2,2*a2,2*a1+a2,a1+2*a2,2*a2+2*a1]
        inflate = lambda i: 3**i # inflate function
    elif mode=="honeycomb":
        g0 = geometry.honeycomb_lattice_C6()
#        g0 = geometry.kagome_lattice() #honeycomb_lattice_C6()
#        g0 = geometry.honeycomb_lattice()
        g0 = supercell.target_angle_volume(g0,angle=1./3.,volume=1,
                                           same_length=True)
#        g0 = g0.get_supercell([2,2])
#        g0.dimensionality = 0
#        g0 = g0.clean()
        r = g0.r
        a1 = g0.a1
        a2 = g0.a2 
        vs = [a1,a2]
        inflate = lambda i: 2**i # inflate function
    else: raise
    for i in range(n): # loop over replicas
        r0 = r.copy() # copy
        for a in vs:
            r = np.concatenate([r,r0[:]+inflate(i)*a]) # add positions
    go = g.copy()
    go.dimensionality = 0
    go.r = r
    go.r2xyz()
    return go # return positions

