"""Amorphous geometries, and the Chern insulator of Agarwala and Shenoy on them

A. Agarwala and V. B. Shenoy, Topological Insulators in Amorphous Systems,
Phys. Rev. Lett. 118, 236402 (2017), arXiv:1701.00374 (titled "Topological
Insulators in Random Lattices" on arXiv)

The sites are placed at random, with a uniform distribution and no
correlation between them, in a square, so there is no lattice at all. Each
site hosts two orbitals, and the hopping between two sites is a 2x2 matrix
that decays exponentially with their distance, up to a cutoff R, and depends
on the direction of the bond, which is what makes the model a Chern
insulator (class A) in a window of the mass M and of the density of sites.
The onsite energy and the hopping are those of the class A row of Table I of
the paper, with the hopping t(r) = e exp(-r) for r <= R, which is 1 at r = 1:
lengths are in units of the decay length of the hopping.
"""
import numpy as np


def amorphous_lattice(L=24.,density=1.0,seed=None):
    """Finite geometry with sites placed at random, uniformly and without
    correlation between them, in an L x L square centered at the origin.

    - L: side of the square, in units of the decay length of the hopping
      of amorphous_chern_hamiltonian
    - density: number of sites per unit area, round(density*L*L) sites
    - seed: seed of the random number generator, the same seed gives the
      same sample
    """
    from ..geometry import Geometry
    n = int(round(density*L*L)) # number of sites
    if n<3:
        raise ValueError("an amorphous sample needs at least three sites, "
                +"and density*L*L = "+str(density*L*L))
    rng = np.random.default_rng(seed) # random number generator
    g = Geometry() # create the geometry
    g.dimensionality = 0 # finite system
    g.r = np.zeros((n,3)) # positions
    g.r[:,0:2] = L*rng.uniform(size=(n,2)) - L/2. # uniform in the square
    g.r2xyz() # update x, y and z
    g.has_sublattice = False # no sublattice in an amorphous sample
    g.name = "amorphous"
    return g


def amorphous_chern_generator(M=-0.5,t2=0.25,lam=0.5,R=4.):
    """Matrix generator of the class A Hamiltonian of Agarwala and Shenoy,
    for g.get_hamiltonian(mgenerator=...,spinful_generator=True).

    The two orbitals of each site take the place of the spin. For two sites
    at distance r in the direction theta (from the first to the second) the
    hopping is t(r) T(theta), with t(r) = e exp(-r) up to r = R and

        T = 1/2 [[-1+t2, -i exp(-i theta) + lam (sin^2(theta) (1+i) - 1)],
                 [-i exp(i theta) + lam (sin^2(theta) (1-i) - 1), 1+t2]]

    and the onsite energy is [[2+M, (1-i) lam], [(1+i) lam, -(2+M)]].

    - M: mass, the parameter that drives the topological transition
    - t2: breaks the particle-hole symmetry of the spectrum
    - lam: mixes the two orbitals, and breaks inversion symmetry
    - R: range of the hopping
    """
    onsite = np.array([[2.+M,(1.-1j)*lam],[(1.+1j)*lam,-(2.+M)]])
    def generator(r1,r2):
        dx = r2[None,:,0] - r1[:,None,0] # x of the bond from site 1 to 2
        dy = r2[None,:,1] - r1[:,None,1] # y of the bond from site 1 to 2
        r = np.sqrt(dx**2 + dy**2) # length of the bond
        same = r<1e-7 # the site itself, where the onsite energy goes
        t = np.e*np.exp(-r)*(r<=R) # radial part of the hopping
        t[same] = 0. # no hopping of a site to itself
        theta = np.arctan2(dy,dx) # direction of the bond
        s2 = np.sin(theta)**2
        m = np.zeros((2*len(r1),2*len(r2)),dtype=np.complex128)
        m[0::2,0::2] = t*(-1.+t2)/2.
        m[1::2,1::2] = t*(1.+t2)/2.
        m[0::2,1::2] = t*(-1j*np.exp(-1j*theta) + lam*(s2*(1.+1j)-1.))/2.
        m[1::2,0::2] = t*(-1j*np.exp(1j*theta) + lam*(s2*(1.-1j)-1.))/2.
        for (i,j) in zip(*np.nonzero(same)): # onsite energy
            m[2*i:2*i+2,2*j:2*j+2] = onsite
        return m
    return generator


def amorphous_chern_hamiltonian(g,M=-0.5,t2=0.25,lam=0.5,R=4.):
    """Class A Chern insulator of Agarwala and Shenoy on a finite geometry,
    at half filling (the Fermi energy is at zero).

    The two orbitals of each site take the place of the spin. With the
    default t2, lam and R the model is topological, with Bott index and
    bulk Chern marker -1, for roughly -2 < M < 1.2 at density 0.6 and
    -2.5 < M < 3.7 at density 1 (Fig. 3 of arXiv:1701.00374), and trivial
    outside. See amorphous_chern_generator for the terms.
    """
    if g.dimensionality!=0:
        raise ValueError("the amorphous Chern insulator is built on a finite "
                "(0d) geometry, and this one has dimensionality "
                +str(g.dimensionality))
    fm = amorphous_chern_generator(M=M,t2=t2,lam=lam,R=R)
    h = g.get_hamiltonian(mgenerator=fm,spinful_generator=True)
    h.set_filling(0.5) # the gap is not centered at zero energy
    return h
