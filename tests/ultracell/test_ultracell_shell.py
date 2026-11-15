"""The partly filled shell of a gapless paired state on the ultracell.

A Nambu mean field at a filling whose pairing dies (here an s-wave state
under a Zeeman field) has no gap on the mesh: its charge is a staircase in
the chemical potential, and the requested charge falls inside a jump, the
shell of levels at zero energy. The ultracell fills that shell in part
with scftk.superscf.ShellFilling, the object of "ed": the occupations of
the shell, the change of the band energy and the shift of the Fermi level
are computed once, there, and the ultracell only adds the change of the
density matrix at the entries the mean field reads, with the vectors of
the shell on the Nambu orbitals of the supercell. The levels of the shell
are eigenstates of the ultracell matrix, so with a cut they are kept
states and the frozen density is not touched. Without it the charge kept
the error of one k-shell, here 0.4 states of a doublet at kS = +-1/3.
"""

import numpy as np
import pytest

from pyqula import geometry, hamiltonians
from pyqula.scftk import superscf
from pyqula.scftk.densitydensity import get_dm, Vinteraction
from pyqula.ultracell import Ultracell
from pyqula.ultracelltk.basis import FoldedBasis, dense
from pyqula.ultracelltk.meanfield import SparseBackend

N = 6 # cells of the supercell
NK = 3 # mesh of the ultracell
STATES = 2*N*NK # electron states on the mesh
# 11.4 of the 36 electron states of the mesh: the Fermi search rounds
# them to 11, the first state of a doublet at kS = +-1/3, which then has to
# hold 0.4 electrons more
FILLING = 11.4/STATES


def charge(filling):
    """Electrons per supercell at a filling"""
    return filling*2*N


def zeeman_chain(field=0.6,v=0.2,direction=(0.,0.,1.)):
    """A spinful chain under a Zeeman field, with the Nambu degree of
    freedom, and its supercell of N cells under a potential periodic on
    it. The field kills the s-wave state of U=-1, so the mean field is a
    normal state written as a BdG Hamiltonian. Without spin-orbit coupling
    the spectrum does not depend on the direction of the field, while the
    density matrix is complex for a field off the z axis"""
    h = geometry.chain().get_hamiltonian(has_spin=True)
    direction = np.array(direction)/np.linalg.norm(direction)
    h.add_zeeman(field*direction)
    h.setup_nambu_spinor()
    hs = h.get_supercell(N)
    hs.add_onsite(lambda r: v*np.cos(2*np.pi*r[0]/N+0.3))
    return h,hs


def paired_chain():
    """The same chain with no field: U=-2.5 at filling 0.28 gives a paired
    state with a gap"""
    h = geometry.chain().get_hamiltonian()
    h.setup_nambu_spinor()
    hs = h.get_supercell(N)
    hs.add_onsite(lambda r: 0.3*np.cos(2*np.pi*r[0]/N+0.3))
    return h,hs


@pytest.fixture
def scfs(monkeypatch):
    """The SCF objects of the mean fields of the test, in order, which
    hold the charge error and the shell of each loop"""
    seen = []
    result = hamiltonians._mean_field_scf_result
    def spy(scf,*args,**kwargs):
        seen.append(scf)
        return result(scf,*args,**kwargs)
    monkeypatch.setattr(hamiltonians,"_mean_field_scf_result",spy)
    return seen


def same_state(u2,h3,k=(0.37,0.,0.)):
    m2 = dense(u2.hs.get_hk_gen()(np.array(k)))
    m3 = dense(h3.get_hk_gen()(np.array(k)))
    return np.max(np.abs(m2-m3))


@pytest.mark.parametrize("T",[1e-7,2e-2])
def test_gapless_state_with_every_state_kept_is_the_supercell(scfs,T,
        tmp_path,monkeypatch):
    """With every state kept, the charge, the energy and the Fermi level
    are those of "ed" on the supercell with its shell. Before, the
    ultracell ended 0.13 electrons per supercell off, 3e-5 in the energy
    and 4.4e-4 in the Fermi level at T=1e-7. At T=2e-2 the shell holds
    levels of several energies and the fill changes the band energy by
    5e-4, which the total energy has to carry. The spinful "ed" of the
    supercell runs the same loop as the ultracell, which adds the change
    of the band energy in one line for both, so the energy and the Fermi
    level are also compared with the loop of densitydensity, which adds it
    in a line of its own"""
    h,hs = zeeman_chain()
    kw = dict(U=-1.,mf="swave",filling=FILLING,nk=NK,T=T,maxerror=1e-11,
            verbose=0,write=False)
    u2,e2 = Ultracell(h,hs).get_mean_field_hamiltonian(
            return_total_energy=True,**kw)
    h3,e3 = hs.get_mean_field_hamiltonian(return_total_energy=True,
            integration="ed",**kw)
    s2,s3 = scfs
    for scf in (s2,s3):
        assert scf.converged
        assert superscf.pairing_amplitude(scf.hamiltonian)<1e-8 # normal
        assert abs(scf.charge_error)<1e-10
        assert abs(superscf.get_charge_bdg(scf.dm)-charge(FILLING))<1e-10
    assert abs(e2-e3)<1e-9
    assert abs(u2.hs.fermi-h3.fermi)<1e-9
    assert same_state(u2,h3)<1e-8
    assert s2.shell.fired and s3.shell.fired
    assert abs(s2.shell.mu-s3.shell.mu)<1e-6*T
    assert abs(s2.shell.correction-s3.shell.correction)<1e-9
    if T>1e-3: assert abs(s2.shell.correction)>1e-4
    monkeypatch.chdir(tmp_path) # the density loop may write its MF.pkl
    s4 = Vinteraction(hs.copy(),load_mf=False,**kw)
    assert s4.shell.fired
    assert abs(e2-s4.total_energy)<1e-9
    assert abs(u2.hs.fermi-s4.hamiltonian.fermi)<1e-9


@pytest.mark.parametrize("T",[1e-7,1e-3])
def test_gapless_state_with_a_cut_holds_the_charge(scfs,T):
    """Cut at |E|<=1, which freezes the BdG states of the bottom of the
    band: the levels of the shell are kept states, and the fill holds the
    charge to the tolerance, where before it ended 0.13 electrons per
    supercell off. The Hamiltonian returned, at the Fermi level the fill
    moved, holds the charge by itself, and the filled density matrix
    keeps particle-hole symmetry"""
    h,hs = zeeman_chain()
    u2 = Ultracell(h,hs,emax=1.).get_mean_field_hamiltonian(U=-1.,
            mf="swave",filling=FILLING,nk=NK,T=T,maxerror=1e-10,verbose=0,
            write=False)
    scf = scfs[0]
    assert scf.converged
    assert superscf.pairing_amplitude(scf.hamiltonian)<1e-8
    assert abs(scf.charge_error)<1e-10
    assert abs(superscf.get_charge_bdg(scf.dm)-charge(FILLING))<1e-10
    assert scf.shell.fired and abs(scf.shell.mu)>0.
    # the cut leaves states out on both sides, as many frozen as dropped
    ks = np.array(hs.geometry.get_kmesh(nk=NK))
    frozen = sum(np.sum(u2.basis.states(k,frozen=True)[5]) for k in ks)
    kept = sum(np.sum(u2.basis.states(k)[2]) for k in ks)
    n = hs.intra.shape[0]
    assert frozen>0 and kept>0 and kept+2*frozen==n*len(ks)
    # the Fermi level moved by the fill holds the charge by itself
    diagonal = {(0,0,0): (np.arange(n),np.arange(n))}
    dm = SparseBackend(nk=NK,T=T).dm(scf.hamiltonian,diagonal)
    assert abs(superscf.get_charge_bdg(dm)-charge(FILLING))<1e-7
    # before that move the shell has to be filled, and the electron and
    # hole occupations of every site still add up to two, one per Nambu
    # pair (the loop reads the electron ones only, so they are computed
    # again here, on every orbital)
    h0 = scf.hamiltonian.copy()
    h0.shift_fermi(scf.shell.mu)
    dm = SparseBackend(nk=NK,T=T).dm(h0,diagonal,shell=scf.shell)
    assert scf.shell.fired
    occ = np.real(dm[(0,0,0)].diagonal()).reshape(-1,4)
    assert abs(np.sum(occ[:,0:2])-charge(FILLING))<1e-10
    assert np.max(np.abs(occ.sum(axis=1)-2.))<1e-10


@pytest.mark.parametrize("T",[1e-7,2e-2])
def test_the_fill_at_the_entries_is_the_dense_fill(T):
    """On the same Hamiltonian, the chain under the field and the
    potential with no pairing, at the Fermi level of its normal state: the
    density matrix of the ultracell with its shell filled, at every entry
    of three directions, is the one of "ed" with its shell filled, with
    the same shift of the chemical potential and the same change of the
    band energy. At T=2e-2 the shell holds levels of several energies, so
    the change of the energy is not zero, and away from d=0 the fill
    carries the Bloch phases. The field points in a generic direction, so
    the fill is complex and a conjugated fill does not pass"""
    h,hs = zeeman_chain(direction=(0.3,0.5,0.8))
    def level(hh):
        fl = superscf.BdGFermiLevel(hh,FILLING,
                lambda x,f: x.get_fermi4filling(f,nk=NK,T=T))
        return fl,fl.shift(hh,0.) # at the Fermi level of the normal state
    f3,h3 = level(hs.copy())
    ds = [(0,0,0),(1,0,0),(-1,0,0)]
    dm3 = get_dm(h3,ds[1:],nk=NK,T=T,integration="ed",shell=f3.shell)
    assert f3.shell.fired
    f2,h2 = level(hs.copy())
    h2.turn_sparse()
    h2.ultracell_basis = FoldedBasis(h,h2)
    n = h2.intra.shape[0]
    r,c = [x.ravel() for x in np.indices((n,n))]
    b = SparseBackend(nk=NK,T=T)
    dm0 = b.dm(h2,{d: (r,c) for d in ds})
    dm2 = b.dm(h2,{d: (r,c) for d in ds},shell=f2.shell)
    assert f2.shell.fired
    assert abs(superscf.get_charge_bdg(dm0)-charge(FILLING))>0.1
    assert abs(superscf.get_charge_bdg(dm2)-charge(FILLING))<1e-12
    for d in ds:
        assert np.max(np.abs(dm2[d].toarray()-dm3[d]))<1e-11
        fill = (dm2[d]-dm0[d]).toarray()
        assert np.max(np.abs(fill))>1e-3 # it fills
        assert np.max(np.abs(fill.imag))>1e-4 # and the fill is complex
    assert abs(f2.shell.mu-f3.shell.mu)<1e-8*T
    assert abs(f2.shell.correction-f3.shell.correction)<1e-12
    assert abs(f2.shell.fraction-f3.shell.fraction)<1e-10
    if T>1e-3: assert abs(f2.shell.correction)>1e-6 # it does something


# the states whose shell is not filled: the model, its keywords, the cut
# and whether the shell ends armed
UNFILLED = {
    # the pairing lives: the shell is asked and never armed
    "paired": (paired_chain,dict(U=-2.5,filling=0.28),None,False),
    "paired, cut": (paired_chain,dict(U=-2.5,filling=0.28),1.5,False),
    # no pairing, and 10 of the 36 states, a whole number of them, between
    # two doublets: armed, and the charge is held already
    "held, cut": (zeeman_chain,dict(U=-1.,filling=10./STATES),1.,True),
    # no pairing, and 10.4 states, rounded to the same 10: the Fermi level
    # is between the two doublets, so the shell holds no level and the
    # charge keeps the error of 0.4 states
    "no level, cut": (zeeman_chain,dict(U=-1.,filling=10.4/STATES),1.,True),
    }


@pytest.mark.parametrize("case",list(UNFILLED))
def test_a_state_whose_shell_is_not_filled_does_not_change(scfs,monkeypatch,
        capsys,case):
    """A paired state, whose shell is asked at every density matrix and
    never armed, and a state with no pairing whose shell is armed and not
    filled, because the charge is held already or because no level sits at
    zero energy: the loop is, bit for bit, the one with the shell switched
    off, with the same iterations, energy, Fermi level and Hamiltonian"""
    model,kw,emax,ends_armed = UNFILLED[case]
    def run():
        h,hs = model()
        return Ultracell(h,hs,emax=emax).get_mean_field_hamiltonian(
                return_total_energy=True,mf="swave",nk=NK,maxerror=1e-7,
                verbose=0,write=False,**kw)
    armed,fired = [],[]
    start,fill = superscf.ShellFilling.start,superscf.ShellFilling.fill
    def counted_start(self,*args):
        armed.append(start(self,*args))
        return armed[-1]
    def counted_fill(self,*args,**kwargs):
        out = fill(self,*args,**kwargs)
        fired.append(self.fired)
        return out
    monkeypatch.setattr(superscf.ShellFilling,"start",counted_start)
    monkeypatch.setattr(superscf.ShellFilling,"fill",counted_fill)
    u1,e1 = run()
    assert len(armed)>0 and armed[-1]==ends_armed # the shell was asked
    if not ends_armed: assert not any(armed)
    assert not any(fired) and not scfs[0].shell.fired
    assert scfs[0].shell.correction==0. and scfs[0].shell.mu==0.
    paired = superscf.pairing_amplitude(u1.hs)>0.05
    assert paired==(not ends_armed)
    if case.startswith("no level"): # the loop says so
        assert abs(abs(scfs[0].charge_error)-0.4/NK)<1e-9
        assert "no gap on this mesh" in capsys.readouterr().out
    else: assert abs(scfs[0].charge_error)<1e-5
    def never(self,*args):
        start(self,*args)
        return False
    monkeypatch.setattr(superscf.ShellFilling,"start",never)
    u2,e2 = run()
    assert scfs[0].iterations==scfs[1].iterations
    assert e1==e2 and u1.hs.fermi==u2.hs.fermi
    d1 = u1.hs.get_multihopping().get_dict()
    d2 = u2.hs.get_multihopping().get_dict()
    assert sorted(d1)==sorted(d2)
    for d in d1: assert np.array_equal(dense(d1[d]),dense(d2[d]))
