# The tetrahedra of the 3D simplex integrations (dostk/simplexdos.py) are
# built in place, without the list, the concatenation and the copy the
# first construction held at once (about 3 times the array at its peak).
# The order of the tetrahedra enters the blocked sums of the kernels, so the
# new array must be the old one entry by entry, and not only the same set.
import itertools

import numpy as np
import pytest

from pyqula.dostk import simplexdos


def _reference(nk):
    """the first construction, from three meshgrids and one stack per
    permutation"""
    i,j,l = np.meshgrid(np.arange(nk),np.arange(nk),np.arange(nk),
            indexing="ij")
    i = i.ravel(); j = j.ravel(); l = l.ravel()
    def idx(a,b,c): return ((i+a)%nk)*nk*nk + ((j+b)%nk)*nk + (l+c)%nk
    out = []
    for perm in itertools.permutations(range(3)):
        v = [0,0,0]
        verts = [idx(*v)]
        for ax in perm:
            v[ax] = 1
            verts.append(idx(*v))
        out.append(np.stack(verts,axis=1))
    return np.concatenate(out).astype(np.int64)


@pytest.mark.parametrize("nk",[1,2,3,5,8])
def test_the_tetrahedra_are_the_first_construction(nk):
    sim,frac = simplexdos.simplices(3,nk)
    ref = _reference(nk)
    assert sim.dtype==np.int64 and sim.flags.c_contiguous
    assert np.array_equal(sim,ref)
    assert frac==1./(6*nk**3)


def test_every_cube_is_cut_into_six_tetrahedra_of_one_volume():
    """each cube of the 4^3 mesh is cut into six tetrahedra of equal
    volume that share its main diagonal, 000 to 111"""
    nk = 4
    sim,_ = simplexdos.simplices(3,nk)
    r = np.stack(np.unravel_index(sim,(nk,nk,nk)),axis=-1) # (t,vertex,3)
    d = (r - r[:,:1,:] + nk//2) % nk - nk//2 # vertices relative to 000
    assert np.all(np.abs(np.linalg.det(d[:,1:,:].astype(float))) == 1.)
    assert np.all(d[:,3,:]==1) # every path ends at the corner 111
