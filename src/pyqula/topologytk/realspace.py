"""Local Chern marker of a finite system (Bianco-Resta)

R. Bianco and R. Resta, Mapping topological order in coordinate space,
Phys. Rev. B 84, 241106(R) (2011), arXiv:1111.5697

The marker of a site is 2*pi*Im<i|[PXP,PYP]|i>, summed over the orbitals of
the site, divided by the area the site occupies. Bianco and Resta divide by
the area of the unit cell and sum over the orbitals of the cell; a sample
with no lattice has no unit cell, and here each site takes the area of its
Voronoi cell instead, the region of the plane closer to it than to any other
site. For a crystal this is the area of the unit cell divided by the number
of sites in it, so both definitions agree, and for an amorphous sample it is
the natural generalization (it is also what Q. Marsal, D. Varjas and A. G.
Grushin use to normalize the marker on amorphous networks, arXiv:2003.13701,
code at zenodo.3741829).
"""
from .. import filewrite
import numpy as np
from .. import densitymatrix


def real_space_chern(h,operator=None,write=None):
    """Local Chern marker of a finite (0d) Hamiltonian, one value per site.

    The occupied states are those below zero energy, so a Hamiltonian whose
    gap is not at zero has to be shifted first (h.set_filling(0.5)).
    Each value is the marker of the site divided by the area of the site
    (see site_areas), which makes it dimensionless: deep inside a crystal
    it equals the Chern number of the occupied bands. On an amorphous
    sample the sites do not all occupy the same area, and the Chern number
    of a region is the area-weighted average of the marker,
    sum_i c_i A_i / sum_i A_i, with A_i = site_areas(h.geometry).

    - operator: restrict the marker to a subspace, as for h.get_chern()

    Returns the positions and the marker, and writes REAL_SPACE_CHERN.OUT
    unless write=False
    """
    write = filewrite.resolve(write,True) # the call, else the global switch
    if h.dimensionality!=0:
        raise ValueError("the real-space Chern number is only defined for 0d "
                "Hamiltonians")
    X = h.get_operator("xposition").get_matrix()
    Y = h.get_operator("yposition").get_matrix()
    m = h.get_hk_gen()([0.,0.,0.]) # get the matrix
    P = densitymatrix.occupied_projector(m).T # projector on occupied states
    A,B = P@X@P,P@Y@P # define the two operators
    if operator is None: # only the diagonal of the commutator is needed
        C = _diagonal_of_product(A,B) - _diagonal_of_product(B,A)
    elif hasattr(operator,"get_matrix"): # restricted to a subspace
        op = operator.get_matrix()
        C1 = A@op@B - B@A@op # compute the commutator
        C2 = A@B@op - B@op@A # compute the commutator
        C = np.diagonal((C1+C2)/2.) # average
    else:
        raise TypeError("the projector must be an operator with a matrix "
                "representation, or None")
    C = 2*np.pi*np.ravel(np.array(C)).imag # marker of each orbital, times an area
    C = h.full2profile(C) # sum over the orbitals of each site
    C = C/site_areas(h.geometry) # divide by the area of each site
    if write: h.geometry.write_profile(C,name="REAL_SPACE_CHERN.OUT")
    return h.geometry.r,C # return result


def _diagonal_of_product(A,B):
    """Diagonal of the matrix product A@B, without computing the product"""
    return np.sum(np.asarray(A)*np.asarray(B).T,axis=1)


def site_areas(g):
    """Area of the plane that each site of a finite geometry occupies.

    The area of a site is the area of its Voronoi cell, the region of the
    xy plane closer to it than to any other site. The sites at the boundary
    of the sample have cells that are open, or that reach outside the
    sample (outside the convex hull of the sites); those take the mean area
    of the cells of the other sites. Sites stacked on top of each other
    (the same x and y) share their cell equally, so that the areas always
    add up to the area of the sample.

    For a crystal every site gets the area of the unit cell divided by the
    number of sites in it, and for an amorphous sample the area around each
    site, which is what turns the marker of real_space_chern into a
    density. Returns one area per site.
    """
    from scipy.spatial import Voronoi, ConvexHull
    xy = np.array(g.r)[:,0:2] # the marker only uses x and y
    # sites stacked along z project onto the same point and share its cell
    (_,first,index,count) = np.unique(np.round(xy,decimals=6),axis=0,
            return_index=True,return_inverse=True,return_counts=True)
    points = xy[first] # one site of each stack
    index = np.reshape(index,-1) # the stack each site belongs to
    centered = points - np.mean(points,axis=0)
    if len(points)<3 or np.linalg.matrix_rank(centered,tol=1e-6)<2:
        raise ValueError("the area of each site needs sites spread over the "
                "xy plane, but the "+str(len(xy))+" sites of this geometry "
                "lie on a line or on a point")
    hull = ConvexHull(points) # the region the sample covers
    voronoi = Voronoi(points)
    area = np.full(len(points),np.nan) # area of each projected point
    for (i,region) in enumerate(voronoi.point_region):
        vertices = voronoi.regions[region]
        if len(vertices)==0 or -1 in vertices: continue # open cell
        corners = voronoi.vertices[vertices] # corners of the cell
        distance = hull.equations[:,0:2]@corners.T + hull.equations[:,2:3]
        if np.any(distance>1e-8): continue # the cell reaches outside
        area[i] = ConvexHull(corners).volume # in 2d the volume is the area
    inside = np.logical_not(np.isnan(area)) # cells inside the sample
    if np.any(inside): mean_area = np.mean(area[inside])
    else: mean_area = hull.volume/len(points) # a sample with no inner site
    area[np.logical_not(inside)] = mean_area # boundary cells
    return area[index]/count[index] # stacked sites share the cell
