# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Magnons from time-dependent Hartree-Fock, i.e. the spin-flip channel of
# the Bethe-Salpeter equation.
#
# A magnon is the same kind of object as an exciton -- a bound two-particle
# excitation of a mean-field state -- except that its electron and hole
# have opposite spin. Solving the BSE in that block gives the magnon
# dispersion of a magnetically ordered mean field directly, with no
# frequency scan, and it works for a neighbor-shell interaction, which the
# site-basis spin RPA (h.get_magnon_bands(), the default method="rpa")
# cannot represent at all.
#
# The check that any of this is right is the Goldstone theorem: a state
# that orders magnetically without spin-orbit coupling breaks SU(2), so a
# uniform spin rotation costs nothing and the acoustic branch must start
# at exactly zero. Here we watch it do that both for a plain Hubbard U and
# with a nearest-neighbor V1 on top -- and note that the mean field and
# the magnon problem MUST be solved on the same k-mesh, or the Ward
# identity behind that zero is broken and the branch lifts off.

import numpy as np
import matplotlib.pyplot as plt
from pyqula import geometry
from pyqula.meanfield import VJinteraction

nk = 6 # the SAME mesh for the SCF and for the magnons
g = geometry.honeycomb_lattice()

fig,axs = plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,(label,kw) in zip(axs,[("U = 3",dict(U=3.0)),
                              ("U = 3, V1 = 0.5",dict(U=3.0,V1=0.5))]):
    scf = VJinteraction(g.get_hamiltonian(),filling=0.5,mf="antiferro",
                         nk=nk,maxerror=1e-10,mix=0.3,maxite=2000,**kw)
    hmf = scf.hamiltonian # a Neel-ordered honeycomb insulator
    print(label,"| moment =",np.round(hmf.get_vev("sz"),4),
          "| interaction keys =",len(hmf.V))
    # how far this mean field is from an exact zero-energy magnon at Q=0.
    # It tracks the SCF tolerance and nothing else, so it is the thing to
    # look at before believing any dispersion below
    print("     Goldstone residual =","%.2e"%hmf.get_goldstone_residual(nk=nk))
    qs,es = hmf.get_magnon_bands(method="tdhf",nk=nk,nq=30,n=3)
    ax.scatter(qs,es.real,c="black",s=6)
    ax.set_title(label) ; ax.set_xlabel("q-point along the path")
    ax.axhline(0.,c="red",lw=0.7)
axs[0].set_ylabel("magnon energy")
plt.tight_layout()
plt.show()
