"""Cotunneling dI/dV of the Kondo lattice mean field (STM spectrum).

A tip above the Kondo pair (c_j, f_j) tunnels into the conduction electron
with amplitude t_c and, through a spin flip of the local moment, into the
composite f fermion with amplitude t_f. Following Maltseva, Dzero and
Coleman, arXiv:0910.1138 (Eq. 2-2 and Eq. 4), the tunneling operator of the
large-N mean field is

    psi_j,s = t_c c_j,s + t_f f_j,s ,

and at weak tunneling the current gives (Eq. Current of the same paper,
and Eq. dIdV of Figgins and Morr, arXiv:1001.4530)

    dI/dV(eV) = sum_s int dw (-df/dw)(w - eV) rho_psi,s(w) ,
    rho_psi = t_c^2 N_cc + t_f^2 N_ff + 2 t_c t_f N_cf ,

with N_cf = -Im G_cf / pi, in units of 2 pi e^2 rho_tip / hbar. The three
terms are the conduction part, the f part and the interference of the two
paths. For a uniform lattice the three Green functions of the mean field
add up to

    G_psi(k,w) = (t_c + t_f V/(w - lam))^2 / (w - eps_k - V^2/(w - lam))
                 + t_f^2/(w - lam) .

The first term is Eq. GpsiKL of 0910.1138 as printed. The second one is
the bare f level: it cancels the pole of the first term at w = lam, so
that there is no state inside the hybridization gap. Summed over a flat
band it is the last term of Eq. GpsiKLsummed of that paper, 2D/(w - lam)
in units of t_c^2 rho, times t_f^2, a factor the printed equation leaves
out.

Here the spectral function is built from the eigenstates of the converged
mean-field Hamiltonian, one diagonalization per k-point and local
projections on the two orbitals of each Kondo pair, so that no matrix is
inverted per energy and a cell of a few hundred Kondo pairs is affordable.
Energies are measured from the chemical potential of the self-consistent
loop (the loop shifts the Fermi energy to zero)."""
import numpy as np
from scipy import linalg

from ..kpointstk import kmesh as kmeshtk
from . import pairs as pairstk


def _get_pairs_and_check(h):
    """Return the Kondo pairs of h, or raise if h is not a Kondo lattice"""
    return pairstk.get_pairs(h, "the cotunneling dI/dV")


_select_sites = pairstk.select_sites


def _tf_per_site(tf, J, h, sites):
    """Amplitude of the f fermion in the tunneling operator, per site.

    With J=None, tf is the amplitude itself (the convention of Figgins and
    Morr). With J given, tf is the bare amplitude t_f of Maltseva et al., the
    amplitude of the spin-flip operator (sigma.S)c in psi, and J is the
    Coqblin-Schrieffer coupling of the self-consistent loop. The exchange of
    the loop is H_I = (J/2) S.(c^dag sigma c), so J/2 is the Heisenberg
    coupling J' of the paper, and [c, H_I] = (J/2) (sigma.S)c. In the mean
    field [c, H_I] is conj(V) f (V_j enters the Hamiltonian as
    V_j f^dag c + h.c.), so (sigma.S)c is 2 conj(V)/J f and the amplitude
    into the f fermion is 2 tf conj(V_j)/J, the one that
    compositespectrum.composite_amplitude uses for the same operator."""
    tfs = np.full(len(sites), tf, dtype=np.complex128)
    if J is None: return tfs
    if not hasattr(h, "hybridization"):
        raise ValueError("J is given, so the amplitude 2*tf*conj(V)/J needs "
                "the converged hybridization, and this Hamiltonian has no "
                "`hybridization`; use the Hamiltonian returned by "
                "get_mean_field_hamiltonian, or pass J=None")
    if J == 0.0:
        raise ValueError("J=0 makes the amplitude 2*tf*conj(V)/J undefined; "
                "pass the Coqblin-Schrieffer coupling used in the "
                "self-consistent loop")
    V = np.asarray(h.hybridization, dtype=np.complex128)
    return 2.*tfs*np.conjugate(V[sites])/J


def _thermal_kernel(T, de):
    """Samples of -df/dw at spacing de, normalized to unit sum"""
    if T <= 0.0: return None
    n = int(np.ceil(12*T/de))
    x = np.arange(-n, n+1)*de
    k = 1.0/(4*T*np.cosh(x/(2*T))**2)
    return k/np.sum(k)


def cotunneling_didv(h, energies=None, tc=1.0, tf=1.0, J=None, nk=20,
        delta=1e-2, T=0.0, site=None, ne=400, emax=None, return_parts=False):
    """Cotunneling differential conductance dI/dV of a Kondo lattice mean
    field, resolved by Kondo pair.

    The tip tunnels into psi = tc*c + tf*f at the site under it (see the
    module docstring), so that

        dI/dV = tc^2 N_cc + |tf|^2 N_ff + 2 Re(tc conj(tf) N_cf)

    summed over the two spin components, in units of 2 pi e^2 rho_tip/hbar.
    The energy is measured from the chemical potential of the self-consistent
    loop, and positive energies are tunneling into empty states.

    h: converged Hamiltonian from KondoLatticeHamiltonian.
        get_mean_field_hamiltonian (dense, spinful).
    energies: uniformly spaced array of energies (the thermal broadening is
        a convolution on that grid). Default: ne points from -emax to emax,
        with emax given or 1.2 times the largest |eigenvalue| at the first
        k-point plus 10*delta.
    tc: amplitude of the direct tunneling into the conduction electron.
    tf: amplitude into the f fermion. With J=None it is the amplitude of the
        f fermion in psi (Figgins and Morr, t_f). With J given it is the
        bare amplitude of Maltseva et al., the amplitude of (sigma.S)c, and
        the f amplitude is 2*tf*conj(V_j)/J, with V_j the converged
        hybridization of the site and J the Coqblin-Schrieffer coupling of
        the self-consistent loop, twice the Heisenberg coupling of the paper.
    nk: k-mesh for each periodic direction (ignored for a finite system).
    delta: Lorentzian broadening of the levels, in the units of h.
    T: temperature of the thermal broadening, with kB=1 (a convolution with
        -df/dw; it is resolved only if T is larger than the energy spacing).
    site: Kondo pair index, or list of them. Default: every pair.
    return_parts: if True, also return a dictionary with the conduction part
        "cc", the f part "ff" and the interference part "cf", each already
        multiplied by its amplitudes, whose sum is dI/dV.

    Returns (energies, dIdV) with dIdV of shape (number of pairs selected,
    number of energies), or one array when site is a single index, plus the
    dictionary of parts (same shape) when return_parts=True. One
    diagonalization per k-point is done."""
    pairs = _get_pairs_and_check(h)
    sites = _select_sites(site, len(pairs))
    if delta <= 0.0:
        raise ValueError("delta must be positive, the broadening of the "
                "levels; got "+str(delta))
    if T < 0.0:
        raise ValueError("T must not be negative; got "+str(T))
    tfs = _tf_per_site(tf, J, h, sites)
    ci = np.array([pairs[s][0] for s in sites])
    fi = np.array([pairs[s][1] for s in sites])
    hkgen = h.get_hk_gen()
    ks = kmeshtk.kmesh(h.dimensionality, nk=nk)
    nsel = len(sites)
    spectra = None # cc, ff, cf (complex) on the padded energy grid
    for k in ks:
        es, vs = linalg.eigh(np.asarray(hkgen(k))) # one diagonalization
        if spectra is None:
            energies = _energy_grid(energies, es, delta, ne, emax)
            de = (energies[-1]-energies[0])/(len(energies)-1)
            kernel = _thermal_kernel(T, de)
            pad = 0 if kernel is None else (len(kernel)-1)//2
            egrid = energies[0]+de*np.arange(-pad, len(energies)+pad)
            spectra = np.zeros((3, nsel, len(egrid)), dtype=np.complex128)
        # weights of every eigenstate on c and f of each pair, over spin
        wcc = np.zeros((nsel, len(es)))
        wff = np.zeros((nsel, len(es)))
        wcf = np.zeros((nsel, len(es)), dtype=np.complex128)
        for s in (0, 1):
            pc = vs[2*ci+s, :]
            pf = vs[2*fi+s, :]
            wcc += np.abs(pc)**2
            wff += np.abs(pf)**2
            wcf += pc*np.conjugate(pf)
        lor = (delta/np.pi)/((egrid[:, None]-es[None, :])**2+delta**2)
        spectra[0] += wcc@lor.T
        spectra[1] += wff@lor.T
        spectra[2] += wcf@lor.T
    spectra /= len(ks)
    if kernel is not None: # thermal broadening, then drop the padding
        spectra = np.array([[np.convolve(row, kernel, mode="same")
                for row in part] for part in spectra])
        spectra = spectra[:, :, pad:len(egrid)-pad]
    elif pad > 0:
        spectra = spectra[:, :, pad:len(egrid)-pad]
    cc = tc**2*spectra[0].real
    ff = np.abs(tfs)[:, None]**2*spectra[1].real
    cf = 2*(tc*np.conjugate(tfs)[:, None]*spectra[2]).real
    didv = cc+ff+cf
    if site is not None and np.ndim(site) == 0: # one pair, one array
        didv, cc, ff, cf = didv[0], cc[0], ff[0], cf[0]
    if return_parts:
        return energies, didv, {"cc": cc, "ff": ff, "cf": cf}
    return energies, didv


def _energy_grid(energies, es, delta, ne, emax):
    """Uniform energy grid: the given one, or one that covers the bands"""
    if energies is None:
        if emax is None: emax = 1.2*np.max(np.abs(es))+10*delta
        return np.linspace(-emax, emax, ne)
    energies = np.asarray(energies, dtype=float)
    if len(energies) < 2:
        raise ValueError("energies needs at least two points")
    if not np.allclose(np.diff(energies), np.diff(energies)[0],
            rtol=1e-6, atol=1e-12):
        raise ValueError("energies must be uniformly spaced, since the "
                "thermal broadening is a convolution on that grid")
    return energies
