from __future__ import print_function
import numpy as np
from . import geometry
from . import sculpt



def get_geometry(shape="polygon",**kwargs):
    if shape=="polygon":
        return get_polygon_island(**kwargs)
    elif shape=="flower":
        return get_flower_island(**kwargs)
    else: raise



def default_geometry_edges(name="square",nedges=None,geo=None):
    lattice_name = name
    if lattice_name=="honeycomb":
      geometry_builder = geometry.honeycomb_lattice
      if nedges==None: nedges = 6
    elif lattice_name=="square":
      geometry_builder = geometry.square_lattice
      if nedges==None: nedges = 4
    elif lattice_name=="kagome":
      geometry_builder = geometry.kagome_lattice
      if nedges==None: nedges = 3
    elif lattice_name=="lieb":
      geometry_builder = geometry.lieb_lattice
      if nedges==None: nedges = 4
    elif lattice_name=="triangular":
      geometry_builder = geometry.triangular_lattice
      if nedges==None: nedges = 3
    else: raise
    # first create a raw unit cell
    if geo is not None: 
      print("Geometry generator taken from input")
      g = geo # builder is input geometry
    else:
      g = geometry_builder()  # build a 2d unit cell
    return g,nedges # get the geometry



def get_polygon_island(n=5,rot=0.,clean=True,shift=[0.,0.,0.],**kwargs):
    """ Create a 0d polygon island"""
    g,nedges = default_geometry_edges(**kwargs)
    nf = float(n)   # get the desired size, in float
    g = g.supercell(int(7*n))   # create supercell
    g.set_finite() # set as finite system
    g.center() # center the geometry
    # now scuplt the geometry
    g = sculpt.rotate(g,rot) # initial rotation
    # now shift the lattice
    g.x += shift[0]
    g.y += shift[1]
    g.z += shift[2]
    g.xyz2r()
    def polygon(g):
        def f(r): 
            return r[0]>-nf*(np.cos(np.pi/3)+1.)  # function to use as cut
        for i in range(nedges): # loop over rotations, 60 degrees
            g = sculpt.intersec(g,f) # retain certain atoms
            g = sculpt.rotate(g,2.*np.pi/nedges) # rotate 60 degrees
        return g # return geometry
    g = polygon(g) # cut according to the polygon rule
    if clean: # if it is cleaned
      g = sculpt.remove_unibonded(g,iterative=True)  # remove single bonded
    g.center() # center the geometry
    return g



def get_flower_island(r=5.,dr=1.,clean=True,**kwargs):
    """ Create a 0d polygon island"""
    g,nedges = default_geometry_edges(**kwargs)
    nf = float(r)   # get the desired size, in float
    g = g.supercell(int(5*r)) # create supercell
    g.set_finite() # set as finite system
    g.set_origin()
    # now scuplt the geometry
    def flower(ri):
        phi = np.arctan2(ri[1],ri[0]) # angle
        rmax = r + dr*np.cos(nedges*phi) # maximum radius
        ri2 = np.sqrt(ri.dot(ri)) # location
        return ri2<rmax # if it is inside
    g = sculpt.intersec(g,flower) # retain certain atoms
    if clean: # if it is cleaned
      g = sculpt.remove_unibonded(g,iterative=True)  # remove single bonded
    g.center() # center the geometry
    return g



