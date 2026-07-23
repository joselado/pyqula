# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")


from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe
import numpy as np
import matplotlib.pyplot as plt

## Same idea as examples/transport/decay_constant, but with a
## superconducting probe (instead of the default normal-metal STM tip).
## With both the probe and the sample superconducting, the ordinary
## scattering-matrix formula behind LocalProbe.didv no longer applies (it
## has no normal lead left to define a reflection amplitude against), so
## LocalProbe.didv/get_kappa automatically route through the
## Floquet-Keldysh multiple-Andreev-reflection (MAR) current instead --
## see Heterostructure.get_dc_current and the "Multiple Andreev
## reflection" section of the user guide. This is *much* more expensive
## than the normal-probe case, so this example uses a coarse energy grid
## and a modest Floquet sideband cutoff (nmax_max) ##

g = geometry.chain()
h = g.get_hamiltonian()
h.shift_fermi(1.) # shift the chemical potential
D = 0.1 # superconducting gap of the sample
h.add_swave(D) # pairing gap of the sample

Dtip = 0.1 # superconducting gap of the probe (STM tip)
lead = geometry.chain().get_hamiltonian()
lead.shift_fermi(1.)
lead.add_swave(Dtip) # the probe lead is itself superconducting

lp = LocalProbe(h,lead=lead,delta=1e-3) # superconducting local probe
lp.T = 0.3 # reference transparency
kwargs = dict(nmax=4,nmax_max=12,tol=5e-2) # keep the Floquet sideband sweep cheap

Dsum = D+Dtip # onset of single-quasiparticle transport (MAR order n=1)
es = np.linspace(1.05*Dsum,1.6*Dsum,7) # energies above the SIS gap sum
ts = [lp.didv(energy=e,**kwargs) for e in es] # dIdV (method="auto" -> "keldysh")
ks = [lp.get_kappa(energy=e,**kwargs) for e in es] # decay rate from the MAR current

plt.subplot(121)
plt.plot(es/Dsum,ts,marker="o")
plt.xlabel("Energy/$(\\Delta+\\Delta_{tip})$") ; plt.ylabel("dIdV")
plt.subplot(122)
plt.plot(es/Dsum,ks,marker="o")
plt.xlabel("Energy/$(\\Delta+\\Delta_{tip})$") ; plt.ylabel("$\\kappa/\\kappa_N$")
plt.tight_layout()
plt.show()
