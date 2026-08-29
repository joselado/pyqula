# X-wave (p-, d-, f-, g-, i-wave) collinear magnets on a lattice.
#
# These are the tight-binding models of Ezawa, "Third-order and fifth-order
# nonlinear spin-current generation in g-wave and i-wave altermagnets, and
# perfectly nonreciprocal spin-current in f-wave magnets", arXiv:2411.16036
# (Phys. Rev. B 111, 125420 (2025)). Every one of them is a spin-independent
# nearest-neighbour kinetic term plus a spin-splitting term
#
#   H_X(k) = eps(k) sigma_0 + J F_X(k) sigma_z
#
# whose form factor F_X(k) is a real trigonometric polynomial that reduces,
# near Gamma, to the X-wave harmonic
#
#   p: kx                              d: kx ky
#   f: kx (kx^2 - 3 ky^2)              g: kx ky (kx^2 - ky^2)
#   i: kx ky (3 kx^2 - ky^2)(kx^2 - 3 ky^2)
#
# (Ezawa's Eqs. (2)-(6)). Because F_X changes sign under the point-group
# operation that exchanges the two spin channels, the net magnetization
# vanishes while the two spin Fermi surfaces are split -- altermagnetism for
# even X, and an odd-parity "p/f-wave magnet" for odd X.
#
# Why not just put a cos(n theta) factor on the nearest-neighbour bonds (the
# thing specialhamiltoniantk/altermagnets.py's dead `n` argument gestures at):
# on a triangular or honeycomb lattice the six nearest-neighbour directions all
# sit at cos(6 theta) = 1, so an l=6 form factor built that way degenerates into
# a uniform spin-dependent bandwidth -- a ferromagnet, not an i-wave
# altermagnet. A genuine i-wave form factor needs bonds beyond the first shell,
# which is exactly what Ezawa's product of six sines encodes.
#
# The sine products are expanded into real-space hoppings *exactly* here (see
# _TrigPoly): sin(k.d) = (e^{i k.d} - e^{-i k.d})/(2i), so a product of n sines
# is a sum of 2^n exponentials, each of which is a hopping at the lattice vector
# sum_j s_j d_j. No Fourier interpolation on a mesh and no truncation is
# involved, and the resulting Hamiltonian reproduces F_X(k) to machine precision
# at every k.
#
# Normalization: the fetched text of the paper garbles some of its prefactors,
# so each one here is fixed instead by requiring the continuum expansion to
# reproduce Ezawa's Eqs. (2)-(6) exactly. That derivation is redone as a test in
# tests/xwave/test_xwave_models.py (a series expansion of the form factor about
# Gamma against the target harmonic), so the constants below are checked rather
# than quoted.
import numpy as np

from .. import geometry
from .. import multicell


class _TrigPoly():
    """A real trigonometric polynomial of the Cartesian momentum, held as
    {Cartesian lattice vector -> complex coefficient} with the convention

        F(k) = sum_R c_R exp(i k.R)

    which is precisely how pyqula stores a Bloch Hamiltonian (see
    geometrytk/bloch.py: the phase of the hopping at integer direction d is
    exp(2 pi i k_red.d) = exp(i k_cart.R_cart)). Products and sums of these
    are exact, so a form factor assembled from sin/cos factors comes out as an
    exact, finite set of hoppings."""

    def __init__(self,terms=None):
        self.terms = dict() if terms is None else dict(terms)

    def __mul__(self,other):
        if not isinstance(other,_TrigPoly): # scalar
            return _TrigPoly({R:c*other for (R,c) in self.terms.items()})
        out = dict()
        for (R1,c1) in self.terms.items():
            for (R2,c2) in other.terms.items():
                R = tuple(np.round(np.array(R1)+np.array(R2),10))
                out[R] = out.get(R,0.)+c1*c2
        return _TrigPoly(out)

    __rmul__ = __mul__

    def __add__(self,other):
        out = dict(self.terms)
        for (R,c) in other.terms.items(): out[R] = out.get(R,0.)+c
        return _TrigPoly(out)

    def __sub__(self,other): return self + (-1.)*other

    def clean(self,tol=1e-10):
        """Drop numerically vanishing terms (the sine products cancel most of
        the 2^n exponentials against one another)"""
        return _TrigPoly({R:c for (R,c) in self.terms.items()
                          if abs(c)>tol})


def _sin(d):
    """sin(k.d) as a _TrigPoly"""
    d = tuple(np.round(np.array(d,dtype=np.float64),10))
    md = tuple(-np.array(d))
    return _TrigPoly({d:1./(2.j), md:-1./(2.j)})


def _cos(d):
    """cos(k.d) as a _TrigPoly"""
    d = tuple(np.round(np.array(d,dtype=np.float64),10))
    md = tuple(-np.array(d))
    return _TrigPoly({d:0.5, md:0.5})


def _prod(polys):
    """Product of a list of _TrigPoly"""
    out = _TrigPoly({(0.,0.,0.):1.+0.j})
    for p in polys: out = out*p
    return out


# The three nearest-neighbour directions of pyqula's triangular lattice
# (geometrytk/lattices.py:triangular_lattice, after its rotate_a2b puts a1
# along x), at 0, 60 and 120 degrees, and the three next-nearest ones at 30,
# 90 and 150 degrees and length sqrt(3). Ezawa's f-wave uses the first set,
# his i-wave the product of both.
_NN_TRI = [np.array([1.,0.,0.]),
           np.array([0.5,np.sqrt(3.)/2.,0.]),
           np.array([-0.5,np.sqrt(3.)/2.,0.])]
_NNN_TRI = [np.array([1.5,np.sqrt(3.)/2.,0.]),
            np.array([0.,np.sqrt(3.),0.]),
            np.array([-1.5,np.sqrt(3.)/2.,0.])]

_EX = np.array([1.,0.,0.])
_EY = np.array([0.,1.,0.])


def _form_factor(wave):
    """Return (lattice name, form factor as a _TrigPoly) for an X-wave.

    The prefactors are those that make the small-k expansion equal Ezawa's
    Eqs. (2)-(6) exactly; tests/xwave/test_xwave_models.py rederives them."""
    if wave=="p": # J kx sigma_z, square lattice
        return "square",_sin(_EX)
    elif wave=="d": # J kx ky sigma_z, square lattice
        return "square",_sin(_EX)*_sin(_EY)
    elif wave=="f": # J kx (kx^2-3 ky^2) sigma_z, triangular lattice
        return "triangular",-4.*_prod([_sin(d) for d in _NN_TRI])
    elif wave=="g": # J kx ky (kx^2-ky^2) sigma_z, square lattice
        return "square",-2.*_sin(_EX)*_sin(_EY)*(_cos(_EX)-_cos(_EY))
    elif wave=="i": # J kx ky (3kx^2-ky^2)(kx^2-3ky^2) sigma_z, triangular
        return "triangular",(16./(3.*np.sqrt(3.)))*_prod(
                [_sin(d) for d in _NN_TRI+_NNN_TRI])
    else: raise ValueError("Unknown wave "+str(wave)+
            ", expected one of p, d, f, g, i")


def _lattice(name):
    """Geometry for a host lattice name"""
    if name=="square": return geometry.square_lattice()
    elif name=="triangular": return geometry.triangular_lattice()
    else: raise ValueError("Unknown lattice "+str(name))


def _integer_direction(g,R,tol=1e-6):
    """Integer lattice direction (n1,n2,0) of a Cartesian vector R, as
    pyqula keys its hoppings. Raises if R is not a lattice vector: that would
    mean the form factor was written with bond vectors that do not belong to
    the host lattice, which is a modelling error rather than a numerical one."""
    A = np.array([g.a1[0:2],g.a2[0:2]]).T # columns are the lattice vectors
    n = np.linalg.solve(A,np.array(R)[0:2])
    ni = np.round(n).astype(int)
    if np.max(np.abs(n-ni))>tol: raise ValueError(
        "The form factor contains the vector "+str(R)+", which is not a "
        "lattice vector of the host lattice")
    return (int(ni[0]),int(ni[1]),0)


def xwave_magnet(wave="d",J=0.3,t=-1.):
    """Tight-binding X-wave collinear magnet of Ezawa, arXiv:2411.16036.

    Parameters
    ----------
    wave : one of "p", "d", "f", "g", "i". Sets both the spin-splitting form
        factor and its host lattice (square for p/d/g, triangular for f/i --
        the paper's choice, and the one whose point group the form factor
        actually respects).
    J : amplitude of the spin splitting, the J of H = eps(k) + J F_X(k) sigma_z.
    t : nearest-neighbour hopping. The default t = -1 puts the band *minimum*
        at Gamma, matching the hbar^2 k^2/2m kinetic term Ezawa expands around;
        pass t = +1 for pyqula's usual sign, which puts the maximum there
        instead. Nothing about the X-wave selection rule depends on this --
        it is a symmetry statement -- but the low-filling continuum limit
        does.

    Returns a spinful multicell Hamiltonian. The two spin channels are exactly
    decoupled (there is no spin-orbit coupling anywhere in these models), which
    is the point: the nonlinear spin currents of arXiv:2411.16036 exist without
    any SOC at all.

    p- and f-wave are odd under k -> -k, so their form factors are odd
    trigonometric polynomials and the corresponding spin-dependent hoppings
    come out *imaginary*. That is correct, not a bug: those are odd-parity
    magnets rather than altermagnets, and the Hamiltonian is still Hermitian
    because c_{-R} = conj(c_R) holds term by term.
    """
    name,F = _form_factor(wave)
    F = F.clean()
    g = _lattice(name)
    h = g.get_hamiltonian(has_spin=True,is_multicell=True,tij=_hopping(t))
    h = h.get_multicell()
    sz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)
    d = multicell.get_hopping_dict(h) # {(n1,n2,n3): matrix}
    d = {k:np.array(v,dtype=np.complex128) for (k,v) in d.items()}
    for (R,c) in F.terms.items():
        key = _integer_direction(g,R)
        m = J*c*sz
        if key in d: d[key] = d[key] + m
        else: d[key] = m
    from ..multihopping import MultiHopping
    h.set_multihopping(MultiHopping(d))
    return h


def _hopping(t):
    """Spin-independent nearest-neighbour hopping generator"""
    def ft(r1,r2):
        dr = r1-r2
        if 0.9<dr.dot(dr)<1.1: return t
        return 0.0
    return ft


def pwave_magnet(J=0.3,**kwargs): return xwave_magnet(wave="p",J=J,**kwargs)
def dwave_altermagnet(J=0.3,**kwargs): return xwave_magnet(wave="d",J=J,**kwargs)
def fwave_magnet(J=0.3,**kwargs): return xwave_magnet(wave="f",J=J,**kwargs)
def gwave_altermagnet(J=0.3,**kwargs): return xwave_magnet(wave="g",J=J,**kwargs)
def iwave_altermagnet(J=0.3,**kwargs): return xwave_magnet(wave="i",J=J,**kwargs)
