"""mode="trace" of the triangle susceptibility is the mean of the diagonal.

The simplex sum is linear in the products of the matrix elements
M^ii_nm(k), so chi_triangle takes the mean over the sites before the
kernel: one component instead of one per site. That has to give the mean
of mode="diagonal", which keeps one component per site, to rounding. We
check it with both kernels, at T = 0 and at finite T (also with delta < T,
where the pieces that hold a resonance are corrected frequency by
frequency), at zero broadening, on a supercell with spin-degenerate
multiplets, and with the eigenvectors recomputed at every pass, where the
trace makes one pass over the matrix elements whatever the number of
sites.
"""
import numpy as np
import pytest

from pyqula import algebra
from pyqula import geometry
from pyqula import operators
from pyqula.chitk import chiAB
from pyqula.chitk import simplexchi


def _supercell():
    g = geometry.honeycomb_lattice().get_supercell(2)
    h = g.get_hamiltonian(has_spin=True) # spin-degenerate multiplets
    h.add_onsite(0.3)
    return h


def _rel(a,b):
    return np.max(np.abs(a-b))/np.max(np.abs(b))


@pytest.mark.parametrize("kernel",["sources","simplices"])
@pytest.mark.parametrize("delta,T",[(1e-2,0.),(0.,0.),(1e-2,1e-2),
                                    (2e-3,1e-2),(0.,1e-2)])
def test_trace_is_the_mean_of_the_diagonal(kernel,delta,T,monkeypatch):
    monkeypatch.setattr(simplexchi,"default_kernel",kernel)
    h = _supercell()
    kw = dict(q=[0.21,0.07,0.],energies=[0.,0.5],nk=4,delta=delta,T=T,
              A="sx",B="sx",integration="triangle")
    diag = np.array(chiAB.chiAB_q(h,mode="diagonal",**kw)[1])
    trace = np.array(chiAB.chiAB_q(h,mode="trace",**kw)[1])
    assert diag.shape==(2,len(h.geometry.r))
    assert trace.shape==(2,)
    ref = np.mean(diag,axis=1)
    assert np.max(np.abs(ref)) > 1e-3
    assert _rel(trace,ref) < 1e-12


def test_trace_makes_one_pass_with_the_eigenvectors_recomputed(monkeypatch):
    """with max_memory_gb too small to keep the eigenvectors or more than
    one component, every pass over the matrix elements diagonalizes k and
    k+q again: in the source kernel, which passes once per block of
    components, mode="diagonal" passes more than once and mode="trace"
    passes once (the reference kernel passes once per block of
    frequencies in both modes)"""
    monkeypatch.setattr(simplexchi,"max_memory_gb",1e-9)
    h = _supercell()
    ni = len(h.geometry.r)
    sx = algebra.todense(h.get_operator("sx").get_matrix())
    ops = np.array([algebra.todense(operators.index(h,n=[i]).get_matrix())@sx
                    for i in range(ni)])
    nk = 4
    kw = dict(q=[0.21,0.07,0.],energies=[0.,0.5],nk=nk,delta=1e-2,T=0.,
              pAs=ops,pBs=ops,counted=True,kernel="sources")
    _,diag,ndiag = simplexchi.chi_triangle(h,mode="diagonal",**kw)
    _,trace,ntrace = simplexchi.chi_triangle(h,mode="trace",**kw)
    assert _rel(trace,np.mean(diag,axis=1)) < 1e-12
    assert ndiag > ntrace
    assert ntrace == 2*nk**2 + 2*nk**2 # the bands, then one pass
