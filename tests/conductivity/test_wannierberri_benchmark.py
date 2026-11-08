"""External benchmark of the second-order responses against WannierBerri
(https://wannier-berri.org, version 26.10): the Berry curvature dipole of
nonlineardc.berry_curvature_dipole against BerryDipole_FermiSea, and the
shift current of photocurrent.shift_current against ShiftCurrent, on the
same tight-binding models written as a Wannier90 _tb.dat file.

The file carries pyqula's numbers unchanged: lattice vectors and orbital
positions in Angstrom stand for lengths in units of pyqula's length a,
hoppings in eV for energies in units of t, so that a = 1 Angstrom and t =
1 eV. The 2D models get a third lattice vector (0,0,c) with no hopping
along it, and c, an arbitrary number, must drop out of the conversion.
The positions enter as the diagonal of the position matrix at R = 0,
which WannierBerri subtracts again (convention_II_to_I), so both codes
work in the atomic gauge with no extra position-matrix term.

Conversion of the units, both into pyqula's e^3 a^(3-d)/(hbar t):

  - Berry curvature dipole. WannierBerri returns the dimensionless
    D_bd = int d^3k/(2 pi)^3 f d_b Omega_d, per volume A c, at T = 0 on a
    grid of Fermi energies. pyqula's D_b = <f d_b Omega_xy> is per area A,
    with dimension of length, so D_b = c D_bz (c in Angstrom). The finite
    temperature is the convolution of the T = 0 result with -f'(E - mu),
    exact for a Fermi-sea sum on a fixed mesh.
  - Shift current. WannierBerri returns sigma_abc in SI units, A/V^2 in
    3D, with the positive elementary charge e in the prefactor e^3 (its
    factors.factor_shift_current), while pyqula's carriers have q = -1.
    Both use J_a = (1/2) sigma_abc E0_b E0_c^* for E(t) = Re E0 exp(-i w t)
    (Sipe's E(w) = E0/2 and j = 2 sigma E(w) E(-w)). The pyqula unit at
    a = 1 Angstrom and t = 1 eV is e^3 a/(hbar t) = (e^2/hbar) Angstrom/V
    in 2D, so sigma = -c sigma_WB/(e^2/hbar), with e^2/hbar in A/V, c in
    Angstrom and the minus sign from q^3 = -e^3. pyqula's own sign is
    pinned to q = -1 by the real-time propagation in test_photocurrent.py.

The two shift currents use different broadenings: pyqula takes
w -> w + i delta in both field frequencies and subtracts the pole eta/delta,
WannierBerri puts a Lorentzian of width delta on the energy conservation of
Sipe's formula. They differ at order delta, and the difference is largest
where the spectrum is small, since the Lorentzian tail of the absorption
edge dominates there: on the gapped honeycomb lattice at delta = 0.05 they
agree to 1 to 3 % for w = 1.3 to 2.0 but differ by 12 % at w = 3.0 and by
about 30 % at w = 4.2. The comparison is therefore made below w = 2.5, and
the difference is checked to halve with delta, so that both codes share
the delta -> 0 limit.

Skipped when wannierberri is not importable (it is not a dependency of
pyqula).
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.conductivitytk import kubo
from pyqula.conductivitytk import nonlineardc
from pyqula.conductivitytk import photocurrent

wb = pytest.importorskip("wannierberri")

from scipy.constants import elementary_charge, hbar


s0 = np.eye(2,dtype=np.complex128)
sx = np.array([[0.,1.],[1.,0.]],dtype=np.complex128)
sy = np.array([[0.,-1j],[1j,0.]],dtype=np.complex128)
sz = np.array([[1.,0.],[0.,-1.]],dtype=np.complex128)

C = 3. # third lattice vector of the 2D models, in Angstrom, drops out


def _tilted_dirac(m=0.4,t=0.3,mu=0.6):
    """Shibata's tilted massive Dirac lattice model, as in
    test_nonlinear_dc.py, H = t sin kx s0 + sin kx sx + sin ky sy
    + [m + 2 - cos kx - cos ky] sz, with the chemical potential at mu"""
    h = geometry.square_lattice().get_hamiltonian(has_spin=True)
    tx = -0.5j*t*s0 - 0.5j*sx - 0.5*sz
    ty = -0.5j*sy - 0.5*sz
    dd = {(0,0,0):(m+2.)*sz,(1,0,0):tx,(-1,0,0):np.conjugate(tx).T,
          (0,1,0):ty,(0,-1,0):np.conjugate(ty).T}
    h.set_multihopping(MultiHopping(dd))
    h.shift_fermi(-mu)
    return h


def _gapped_honeycomb(m=0.5):
    """Spinless honeycomb lattice with a staggered onsite mass, x along the
    armchair direction: D3h and time reversal, gap 2m"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    h.add_sublattice_imbalance(m)
    return h


def _write_tb(h,fname):
    """Write h as a Wannier90 _tb.dat file, in the order of WannierBerri's
    own system_tb.write_tb_file: for every R the lines "m n Re Im" of
    H_mn(R) = <m0|H|nR>, n the outer loop, then the position matrix, only
    its diagonal at R = 0, the orbital positions"""
    dd = h.get_multihopping().get_dict()
    n = h.intra.shape[0]
    g = h.geometry
    lattice = np.array([g.a1,g.a2,[0.,0.,C]],dtype=np.float64)
    lattice[:2,2] = 0.
    pos = kubo._orbital_positions(h,n)
    Rs = sorted(dd.keys())
    with open(fname,"w") as f:
        f.write("pyqula tight-binding model\n")
        for v in lattice: f.write(" ".join("%.16f"%x for x in v)+"\n")
        f.write("%d\n%d\n"%(n,len(Rs)))
        f.write(" ".join("1" for _ in Rs)+"\n") # no Wigner-Seitz weights
        for R in Rs: # the hoppings
            M = np.asarray(dd[R],dtype=np.complex128)
            f.write("\n%d %d %d\n"%tuple(R))
            for j in range(n):
                for i in range(n):
                    f.write("%d %d %.16e %.16e\n"%(i+1,j+1,M[i,j].real,
                        M[i,j].imag))
        for R in Rs: # the position matrix
            f.write("\n%d %d %d\n"%tuple(R))
            for j in range(n):
                for i in range(n):
                    if i==j and tuple(R)==(0,0,0): x = pos[i]
                    else: x = np.zeros(3)
                    f.write("%d %d "%(i+1,j+1)
                            +" ".join("%.16e 0.0"%xi for xi in x)+"\n")


def _system(h,path):
    """WannierBerri system of h, after checking its bands against pyqula's
    at random reduced momenta (a mistake k -> -k shows up there with the
    tilted Dirac model; a transposed H(k) would flip the dipole)"""
    fname = str(path/"model_tb.dat")
    _write_tb(h,fname)
    system = wb.system.System_R.from_tb_dat(tb_file=fname,berry=True)
    hk = h.get_hk_gen()
    for k in np.random.default_rng(1).random((4,3)):
        k[2] = 0.
        e1 = wb.evaluate_k(system,k=k,quantities=["energy"])
        e2 = np.linalg.eigvalsh(np.asarray(hk(k)))
        assert np.allclose(e1,e2,atol=1e-12)
    return system


def _run(system,nk,calculator,path):
    """One WannierBerri calculator on the uniform nk x nk x 1 grid, serial,
    without symmetries, its result files written under path"""
    grid = wb.Grid(system,NK=(nk,nk,1),NKFFT=(nk,nk,1),use_symmetry=False)
    res = wb.run(system,grid=grid,calculators={"x":calculator},
            parallel=False,use_irred_kpt=False,symmetrize=False,
            fout_name=str(path/"wb"))
    return res.results["x"].data


def test_bcd_matches_wannierberri(tmp_path,monkeypatch):
    """D_a of the tilted Dirac model at mu = 0.6 and T = 0.05 agrees with
    WannierBerri's Fermi-sea dipole to 1 %. Both are converged at nk=100:
    pyqula's D_x moves by 1e-5 of itself and WannierBerri's by 3e-4 to
    nk=200, where they agree to 0.02 %. The tilt along x gives D_x and the
    mirror y -> -y kills D_y in both"""
    monkeypatch.chdir(tmp_path)
    h = _tilted_dirac()
    T,nk = 0.05,100
    D = nonlineardc.berry_curvature_dipole(h,nk=nk,T=T)
    system = _system(h,tmp_path)
    dE = T/40.
    Ef = np.arange(-0.6,0.6+dE/2.,dE) # mu = 0 after the shift, +-12 T
    d = _run(system,nk,wb.calculators.static.BerryDipole_FermiSea(
            Efermi=Ef,tetra=False),tmp_path) # (Ef,b,d) at T = 0
    w = 0.25/np.cosh(Ef/(2.*T))**2/T # -f'(E), the convolution to T > 0
    Dwb = C*np.array([np.sum(w*d[:,a,2])*dE for a in range(2)])
    assert abs(D[0])>1e-3
    assert abs(Dwb[0]-D[0])<1e-2*abs(D[0])
    assert abs(D[1])<1e-8*abs(D[0]) and abs(Dwb[1])<1e-8*abs(D[0])


def test_shift_current_matches_wannierberri(tmp_path,monkeypatch):
    """The shift current of the gapped honeycomb lattice at w = 1.3, 1.6
    and 2.0 (gap 1.0, the van Hove transition at 2.24) agrees with
    WannierBerri's to 5 % at delta = 0.04 (measured 0.6, 2.1 and 2.5 %),
    with the D3h pattern xxx = -xyy = -yxy = -yyx and the components with
    an odd number of y zero in both. nk=200 is converged: nk=400 moves
    pyqula's xxx by 0.2 % at this delta. The difference is a broadening
    effect, linear in delta: at delta = 0.08 it is twice as large (1.2, 4.1
    and 5.1 %), so the two codes have the same delta -> 0 limit, which a
    wrong prefactor would not give"""
    monkeypatch.chdir(tmp_path)
    h = _gapped_honeycomb()
    nk = 200
    ws = np.array([1.3,1.6,2.0])
    comps = ["xxx","xyy","yxy","yyx","yyy","yxx","xxy","xyx"]
    system = _system(h,tmp_path)
    conv = -C/(elementary_charge**2/hbar) # SI to e^3 a/(hbar t), q = -1
    idx = {"x":0,"y":1,"z":2}
    err = {}
    for delta in [0.04,0.08]:
        (w,s) = photocurrent.shift_current(h,energies=ws,nk=nk,T=0.,
                delta=delta,components=comps)
        d = _run(system,nk,wb.calculators.dynamic.ShiftCurrent(
                Efermi=np.array([0.]),omega=ws,sc_eta=1e-8,kBT=0.,
                smr_fixed_width=delta,smr_type="Lorentzian"),tmp_path)
        swb = {c:conv*d[0,:,idx[c[0]],idx[c[1]],idx[c[2]]] for c in comps}
        ref = np.abs(s["xxx"])
        assert np.min(ref)>1e-2
        for out in [s,swb]: # the D3h pattern in both codes
            for c in ["xyy","yxy","yyx"]:
                assert np.allclose(out[c],-out["xxx"],atol=1e-8*ref)
            for c in ["yyy","yxx","xxy","xyx"]:
                assert np.allclose(out[c],0.,atol=1e-8*ref)
        err[delta] = np.abs(np.real(s["xxx"])-swb["xxx"])/ref
    assert np.all(err[0.04]<5e-2)
    assert np.all(err[0.04]<0.7*err[0.08]) # the difference halves with delta
