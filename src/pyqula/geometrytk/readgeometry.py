from ..geometry import Geometry
import numpy as np


def generate_keep_species(species):
    """Function to choose whether to keep a certain species"""
    if species is None: return lambda x: True # all
    def fkeep(s): return s in species
    return fkeep

def read_xyz(input_file="positions.xyz",species=None):
    """Read an xyz geometry"""
    fkeep = generate_keep_species(species) # function to decide species
    ls = open(input_file).readlines() # read the geometry
    del ls[0] # delete this line
    del ls[0] # delete this line
    rs = [] # empty list
    for i in range(len(ls)): # loop over lines
        l = ls[i].split() # split the line
        if len(l)<4: continue
        s = l[0]
        if fkeep(s): # check if this site should be kept
          x,y,z = float(l[1]),float(l[2]),float(l[3])
          rs.append([x,y,z])
    rs = np.array(rs) # convert to array
    from ..neighbor import neighbor_distances
    g = Geometry() # create geometry
    g.dimensionality = 0
    g.r = rs # positions
    g.r2xyz()
    g.normalize_nn_distance() # set the minimum distance to 1
    return g



