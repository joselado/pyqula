"""The linear spin waves of spinwaves.hp_heisenberg, a Holstein-Primakoff
transformation of a ferromagnetic Heisenberg model with spins S=1.

For a coupling J(d) to the site at the lattice vector d, the magnon energy
is E(k) = sum_d J(d) (1 - cos(2 pi k.d)), zero at k=0, the Goldstone mode of
the ferromagnet. The couplings are given as a function, which on a lattice
of one site per cell reaches beyond the neighboring cells, so a coupling
dropped there shows up as a wrong dispersion.
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula import spinwaves


def shells(ds,js):
    def f(r1,r2):
        d = np.linalg.norm(np.array(r1)-np.array(r2))
        return sum(j*float(abs(d-di)<1e-3) for di,j in zip(ds,js))
    return f


CASES = [("chain",[1.,2.,3.],[1.,0.3,-0.1]),
         ("square",[1.,np.sqrt(2.),2.],[1.,0.3,-0.1])]


@pytest.mark.parametrize("name,ds,js",CASES)
def test_the_magnon_dispersion_has_every_coupling(name,ds,js):
    g = getattr(geometry,{"chain":"chain","square":"square_lattice"}[name])()
    sm = spinwaves.SpinModel(g)
    sm.hp_heisenberg(fun=shells(ds,js))
    hk = sm.hamiltonian.get_hk_gen()
    fun = shells(ds,js)
    cells = [np.array([i,j,0]) for i in range(-4,5) for j in range(-4,5)]
    if g.dimensionality==1: cells = [c for c in cells if c[1]==0]
    def exact(k):
        out = 0.
        for c in cells:
            d = c[0]*np.array(g.a1) + c[1]*np.array(g.a2)
            out += fun(np.zeros(3),d)*(1.-np.cos(2.*np.pi*np.dot(k,c)))
        return out
    for k in [np.zeros(3),np.array([0.13,0.29,0.]),np.array([0.41,0.07,0.])]:
        if g.dimensionality==1: k = np.array([k[0],0.,0.])
        e = np.linalg.eigvalsh(np.array(hk(k)))[0]
        assert abs(e-exact(k))<1e-10
    # and the SzSz onsite term is the sum of the couplings of a site
    ons = np.array(spinwaves.SpinModel(g).szsz(fun=shells(ds,js)))
    total = sum(fun(np.zeros(3),c[0]*np.array(g.a1)+c[1]*np.array(g.a2))
            for c in cells if np.any(c!=0))
    assert abs(ons[0,0]-total)<1e-10
