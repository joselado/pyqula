# Add the root path of the pyqula library
import os ; import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

# Dynamical spin susceptibility of the local moments of the Kondo lattice
# in the mean field of the auxiliary fermions, and with a residual
# interaction in the random phase approximation, as in Chen, Stoudenmire,
# Komijani and Coleman, arXiv:2302.09701.
#
# The moment is S = f^dag sigma f / 2, so a spin flip is a particle and a
# hole of the hybridized bands. The mean-field susceptibility is the product
# of two f propagators,
#   chi(q,w) = 1/N sum_k w-(k) w+(k+q) [1/(w-D) - 1/(w+D)] ,
#   D = E+(k+q) - E-(k) ,
# with w+-(k) the weight of the f fermion in the band E+-(k). It is a
# continuum. The interaction between the particle and the hole is put back
# with a residual interaction U'(q),
#   chi_RPA = 1/(1/chi - U'(q)) ,
# and U' < 0 binds a magnon below the continuum.
#
# That work writes the Kondo term as J' (c^dag sigma c).S, and the loop of
# pyqula takes J = 2 J'. Its two cases are J' = 2 t with U' = -2 t, and
# J' = 0.9 t with U'(q) = -0.75 t sin(q/2).
#
# Sign convention of pyqula: the static response is negative, and the
# spectral weight at positive frequency is +Im chi/pi.

import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian

gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True) # conduction electrons, eps_k = 2 cos k
h = KondoLatticeHamiltonian(hc)

def bands(k, V):
    """Two bands of the half-filled chain and f weight of the upper one"""
    e = 2*np.cos(2*np.pi*k)
    r = np.sqrt((e/2)**2+V**2)
    return e/2-r, e/2+r, (r-e/2)**2/((r-e/2)**2+V**2)

delta = 0.05
nk = 200
ws = np.linspace(0., 6., 301)
for (Jp, U, scale) in [(2.0, -2.0, 1.0),
        (0.9, lambda q: -0.75*np.sin(np.pi*q[0]), 1.5)]:
    h2 = h.get_mean_field_hamiltonian(J=2*Jp, filling=0.5, nk=400,
            mf=([Jp+0j], [0.0]), maxerror=1e-8)
    V = h2.hybridization[0].real*scale
    print("J' =", Jp, ", hybridization V times", scale, "=", V)
    em, ep, wp = bands(np.arange(4000)/4000, V)
    print("direct gap 2V and indirect gap")
    print(2*V, np.min(ep)-np.max(em))
    for q in [0.0, 0.25, 0.5]:
        # T is the temperature of the occupations, far below the gap
        kw = dict(q=[q, 0., 0.], energies=ws, nk=nk, delta=delta, T=1e-3,
                hybridization_scale=scale)
        (es, chi0) = h2.get_moment_spinchi(**kw) # two f propagators
        (es, chi) = h2.get_moment_spinchi(U=U, **kw) # with U'(q)
        # the formula of the two propagators, summed on the same mesh
        ks = np.arange(nk)/nk
        em, ep, wp = bands(ks, V)
        emq, epq, wpq = bands(ks+q, V)
        z = ws[:, None]-1j*delta
        ref = np.mean((1-wp)*wpq/(z-(epq-em))-wp*(1-wpq)/(z+(ep-emq)), axis=1)
        (es, chis) = h2.get_moment_spinchi(**dict(kw, energies=[0.0]))
        Uq = U(np.array([q, 0., 0.])) if callable(U) else U
        print("q/2pi =", q)
        print("  difference to the formula of the two propagators:",
                np.max(np.abs(chi0[:, 0, 0]-ref)))
        print("  maximum of Im chi, mean field: w =",
                ws[np.argmax(chi0[:, 0, 0].imag)])
        print("  maximum of Im chi, RPA:        w =",
                ws[np.argmax(chi[:, 0, 0].imag)])
        # below zero the mean field is unstable to magnetic order
        print("  static denominator 1 - U' chi(q,0):",
                1-Uq*chis[0, 0, 0].real)

# the local response: its integral is <S^- S^+> of one site, 1/4 for free
# f fermions instead of the 1/2 of a spin one half
h2 = h.get_mean_field_hamiltonian(J=4.0, filling=0.5, nk=400,
        mf=([2.0+0j], [0.0]), maxerror=1e-8)
wl = np.linspace(0., 60., 3001)
(es, chil) = h2.get_moment_spinchi(energies=wl, nk=20, delta=delta, T=1e-3)
print("integral of the local spectral weight (1/4 in the mean field)")
print(np.trapezoid(chil[:, 0, 0].imag, wl)/np.pi)

# the map over (q,w) for J' = 2 t, mean field and RPA
qs = np.linspace(0., 1., 41)
out = []
for q in qs:
    (es, chi0) = h2.get_moment_spinchi(q=[q, 0., 0.], energies=ws, nk=nk,
            delta=delta)
    (es, chi) = h2.get_moment_spinchi(q=[q, 0., 0.], energies=ws, nk=nk,
            delta=delta, U=-2.0)
    for (w, a, b) in zip(ws, chi0[:, 0, 0].imag, chi[:, 0, 0].imag):
        out.append([q, w, a, b])
np.savetxt("MOMENT_SPINCHI.OUT", np.array(out))
