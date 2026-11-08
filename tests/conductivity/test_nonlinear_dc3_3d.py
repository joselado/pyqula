"""The third-order nonlinear Drude term in 3D
(src/pyqula/conductivitytk/nonlineardrude.py, nonlineardc order=3).

A one-band cubic model goes through the closed-form derivative of
nonlineardrude._Channel, and its 2x2x2 supercell through the three-index
Kato recursion, on the same set of k-points. The folded bands of the
supercell are degenerate on whole planes of its zone, so the comparison pins
the z index of the recursion, the multiplet trace and the cell volume at the
same time.
"""
import numpy as np

from pyqula import geometry
from pyqula import conductivity
from pyqula.multihopping import MultiHopping
from pyqula.conductivitytk import nonlineardc


LAB4 = [a+b+c+d for a in "xyz" for b in "xyz" for c in "xyz" for d in "xyz"]


def _cubic_metal(c=1.7,mu=-1.2,has_spin=False):
    """One orbital per cell of a tetragonal lattice (a3 of length c), with
    first-neighbor hoppings that differ along x, y and z and second- and
    third-neighbor hoppings coupling z to the plane, so that the band
    disperses along z and the mixed derivatives are finite"""
    g = geometry.cubic_lattice()
    g.a3 = np.array([0.,0.,c])
    h = g.get_hamiltonian(has_spin=has_spin)
    n = 2 if has_spin else 1
    one = np.eye(n,dtype=np.complex128)
    ts = {(1,0,0):-1.,(0,1,0):-0.8,(0,0,1):-0.6,(1,0,1):-0.25,(0,1,1):0.15,
          (1,1,1):-0.1,(1,-1,0):0.2}
    dd = {(0,0,0):-mu*one}
    for (R,t) in ts.items():
        dd[R] = t*one
        dd[tuple(-np.array(R))] = t*one
    h.set_multihopping(MultiHopping(dd))
    return h


def test_supercell_gives_the_same_drude_term_in_3d():
    """The primitive cell at nk=16 and its 2x2x2 supercell at nk=8 are the
    same k-points, so the whole (3,3,3,3) Drude tensor agrees to rounding,
    and the components with z indices are finite"""
    h = _cubic_metal()
    hs = h.get_supercell((2,2,2))
    assert len(hs.geometry.r)==8
    kw = dict(order=3,term="drude",T=0.1,components=LAB4)
    a = nonlineardc.nonlinear_dc_conductivity(h,nk=16,**kw)
    b = nonlineardc.nonlinear_dc_conductivity(hs,nk=8,**kw)
    big = max(abs(a[x]) for x in LAB4)
    assert big>1e-3
    for x in ["zzzz","xxzz","yyzz","xzzz","xyzz"]:
        assert abs(a[x])>1e-3*big,(x,a[x])
    for x in LAB4: assert abs(a[x]-b[x])<1e-12*big,(x,a[x],b[x])


def test_spinful_drude_conductivity_in_3d():
    """nonlinear_drude_conductivity runs in 3D, and its charge channel of
    the spinful model is twice the dc Drude term of the spinless one"""
    h = _cubic_metal()
    hs = _cubic_metal(has_spin=True)
    d = nonlineardc.nonlinear_dc_conductivity(h,order=3,term="drude",nk=12,
            T=0.1,components=["zzzz","xxzz"])
    for (f,b,x) in [("zzz","z","zzzz"),("xzz","x","xxzz")]:
        s = conductivity.nonlinear_drude_conductivity(hs,field=f,current=b,
                channel="charge",nk=12,T=0.1)
        assert abs(s-2.*d[x])<1e-12*abs(d[x]),(x,s,d[x])
