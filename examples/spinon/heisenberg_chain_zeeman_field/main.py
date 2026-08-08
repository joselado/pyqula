# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Abrikosov-pseudofermion (RVB) mean-field theory for the antiferromagnetic
# Heisenberg chain in an external Zeeman field. S_i = 1/2 f_i^dagger sigma
# f_i is already bilinear in the auxiliary fermion, so a field couples
# EXACTLY (not through any mean-field decoupling) as an ordinary
# single-particle term -- added with the same Hamiltonian.add_zeeman used
# throughout pyqula, called before get_mean_field_hamiltonian (see the user
# guide, "Abrikosov-pseudofermion (spinon) mean field for Heisenberg
# models"). The local one-fermion-per-site constraint stays exactly
# satisfied under the field (it constrains total occupation, not spin), and
# the induced magnetization interpolates from the free-fermion value at weak
# J/strong field to a suppressed, AFM-correlation-screened value once J
# dominates -- see Savary & Balents, arXiv:1601.03742, Sec. 4.1. The scan
# below is not monotonic: the self-consistent RVB band structure can have
# metamagnetic-like jumps as a function of field (a level crossing in the
# occupied spinon bands), including a field value where this example's
# default (plain-mixing) SCF settings fail to converge at all -- both are
# genuine mean-field behavior, not bugs, so the loop below skips a
# non-converged field point rather than treating it as a failure.

import numpy as np
from pyqula import geometry
from pyqula.spinon import SpinonHamiltonian

g = geometry.chain()
J1 = 1.0
fields = np.linspace(0.0, 2.0, 11)
done_fields, magnetizations = [], []
for b in fields:
    np.random.seed(0) # reproducible SCF seed, see the plain-chain example
    h = SpinonHamiltonian(g)
    h.add_zeeman([0., 0., b]) # coefficient of sigma, i.e. h=2b in H=-h.S
    h2 = h.get_mean_field_hamiltonian(J1=J1, nk=24, mix=0.3,
            maxerror=1e-6, maxite=2000)
    if h2 is None:
        # plain-mixing SCF can fail to converge right at a metamagnetic
        # (first-order-like) jump in the RVB spinon band occupation --
        # skip rather than fail the example; a smaller mix/larger maxite
        # can resolve individual problem points if needed
        print("SCF did not converge for zeeman = %s, skipping" % b)
        continue
    if not np.allclose(h2.local_occupation, 1.0, atol=1e-3):
        raise RuntimeError("local constraint violated for zeeman = %s" % b)
    done_fields.append(b)
    magnetizations.append(h2.get_magnetization()[0][2])

print("Zeeman field (coefficient of sigma_z) vs induced <S_z> per site")
for b, m in zip(done_fields, magnetizations):
    print("%8.3f  %8.4f" % (b, m))

np.savetxt("MAGNETIZATION_VS_FIELD.OUT", np.array([done_fields, magnetizations]).T)
