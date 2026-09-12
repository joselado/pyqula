# Bug audit 2 -- reproduction scripts

The 211 standalone scripts written while producing [`bug_audit_2.md`](bug_audit_2.md)
and then fixing it -- the eight audit lenses' reproductions, plus the
verification scripts the fourteen fixing agents wrote. Preserved verbatim apart
from one substitution: absolute paths have been replaced by the placeholders
`<repo root>` and `SCRATCH`, so the scripts carry no machine-specific layout and
can be pointed at any checkout. Fill those in before running one.

They are evidence for the findings, not part of the package, and nothing imports
them.

**How to run one.** They are plain scripts, not tests. From the repo root:

```bash
PYTHONPATH=src python3 <paste the script into a file>
```

A few write output files (`BANDS.OUT`, `MULTILDOS/`, `MF.pkl`) into the current
directory -- run them somewhere other than the repo root, since a stray
`MF.pkl` will seed a later SCF (`load_mf=True` is the default).

**They are not a test suite.** Each one prints numbers a human reads; none
asserts. The numbers they printed are the PRE-FIX state, quoted in
`bug_audit_2.md`; 76 of the 80 findings have since been fixed, so re-running one
today will generally print something different -- that is the point. Scripts
whose finding was *cleared* (investigated and found not to be a defect) are
marked as such below.

A script with no annotation was exploratory: it was written along the way and
did not end up backing a specific numbered entry.

---

## `ah_mix.py`

```python
"""attractive_hubbard (scftk/attractive_hubbard_spinless.py) computes its
convergence residual AFTER mixing, so the reported error is (1-mix) times the
true fixed-point residual, and is identically zero for mix=1."""
import numpy as np, os, io, contextlib
from pyqula import geometry
from pyqula.scftk.attractive_hubbard_spinless import attractive_hubbard
from pyqula.sctk.spinless import onsite_delta_vev
from pyqula.superconductivity import get_eh_sector

if os.path.exists("MF.pkl"): os.remove("MF.pkl")
G = -2.0 ; NK = 6

def build():
    ge = geometry.chain()
    h = ge.get_hamiltonian()
    h.remove_spin()
    return h

for mix in (1.0,0.9,0.5,0.1):
    np.random.seed(0)
    h = build()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scf = attractive_hubbard(h,g=G,nk=NK,mix=mix,maxerror=1e-6)
    lines = buf.getvalue().splitlines()
    reported = [float(l.split("=")[1]) for l in lines if l.startswith("Error = ")]
    hh = scf.hamiltonian
    d_in = np.diag(np.array(get_eh_sector(np.array(hh.intra),i=0,j=1)))/G
    d_out = onsite_delta_vev(hh,nk=NK)
    true_res = np.max(np.abs(d_out-d_in))
    print("mix=%4.2f iters=%3d  last REPORTED error=%.4e   TRUE |F(x)-x|=%.4e  ratio=%.4f"
          %(mix,len(reported),reported[-1],true_res,true_res/max(reported[-1],1e-300)))
    print("         returned |Delta| =",np.round(np.abs(d_in*G),8))
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
```

## `ah_mix2.py`

```python
import numpy as np, os, io, contextlib
from pyqula import geometry
from pyqula.scftk.attractive_hubbard_spinless import attractive_hubbard
from pyqula.sctk.spinless import onsite_delta_vev

if os.path.exists("MF.pkl"): os.remove("MF.pkl")
G = -2.0 ; NK = 6
def build():
    h = geometry.chain().get_hamiltonian(); h.remove_spin(); return h

for mix in (1.0,0.9,0.5):
    np.random.seed(0)
    h = build()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        scf = attractive_hubbard(h,g=G,nk=NK,mix=mix,maxerror=1e-6)
    lines=[l for l in buf.getvalue().splitlines() if l.startswith("Error = ")]
    hh = scf.hamiltonian
    print("mix=%4.2f iters=%d  last printed TRUE error=%s"%(mix,len(lines),lines[-1].split("=")[1]))
    print("   intra=\n",np.round(np.array(hh.intra),6))
    d_out = onsite_delta_vev(hh,nk=NK)
    print("   <cc> of returned H:",np.round(d_out,8))
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
```

## `anom_bond.py`

Backs: **L1 SCF/Nambu** — CLEARED: bond (d != 0) anomalous mean field vanishing for an s-wave-paired chain

```python
import numpy as np
from pyqula import geometry
from pyqula import superconductivity as sc
from pyqula.scftk.superscf import get_mf_anomalous, anomalous_term_ij
from pyqula.scftk.spinspin import _build_density_v

g = geometry.chain()
hs = g.get_hamiltonian(has_spin=True)
v = _build_density_v(hs, V1=1.0, U=0.0)
ds = list(v.keys())
print("v keys:",ds)
for k in ds: print("  v[%s]=\n"%(k,),np.round(np.array(v[k]),3))

h = g.get_hamiltonian(has_spin=True); h.turn_nambu(); h.add_swave(0.3); h.shift_fermi(0.7)
dm = h.get_density_matrix(ds=ds,nk=24)
dma10 = {k: np.array(sc.get_eh_sector(m,i=0,j=1)) for k,m in dm.items()}
for k in sorted(dma10, key=str):
    print("dma(0,1)[%s] max=%.5e\n"%(k,np.max(np.abs(dma10[k]))),np.round(dma10[k],5))
mfa = get_mf_anomalous(v,dma10)
for k in sorted(mfa,key=str):
    print("mfa[%s] max=%.5e"%(k,np.max(np.abs(np.array(mfa[k])))))
    print(np.round(np.array(mfa[k]),5))
```

## `anom_sym.py`

Backs: **L1 SCF/Nambu** — CLEARED: superscf.enforce_eh_symmetry_anomalous halving or distorting the anomalous mean field

```python
"""Is enforce_eh_symmetry_anomalous the identity on get_mf_anomalous' output?
If it is, the 2*v[d] prefactor derived analytically survives untouched.
If it changes it, the projection is doing real work (and could halve it)."""
import numpy as np
from pyqula import geometry
from pyqula import superconductivity as sc
from pyqula.scftk.superscf import get_mf_anomalous, enforce_eh_symmetry_anomalous
from pyqula.scftk.spinspin import _build_density_v

g = geometry.chain()
hs = g.get_hamiltonian(has_spin=True)
v = _build_density_v(hs, V1=1.0, U=-2.0)
ds = list(v.keys())

for label,setup in [("swave", lambda h: h.add_swave(0.3)),
                    ("pwave", lambda h: h.add_pairing(delta=0.3,mode="pwave")),
                    ("swave+exchange", lambda h:(h.add_swave(0.3),h.add_exchange([0.2,0.1,-0.3])))]:
    h = g.get_hamiltonian(has_spin=True); h.turn_nambu(); setup(h)
    dm = h.get_density_matrix(ds=ds,nk=12)
    dma10 = {k: sc.get_eh_sector(m,i=0,j=1) for k,m in dm.items()}
    mfa = get_mf_anomalous(v,dma10)
    mfa01,mfa10 = enforce_eh_symmetry_anomalous(mfa)
    for k in sorted(mfa):
        a = np.array(mfa[k]); b = np.array(mfa01[k])
        d = np.max(np.abs(a-b)); s = np.max(np.abs(a))
        if s>1e-12:
            print("%-16s d=%-12s scale=%.4e  |before-after|=%.3e  ratio(after/before)=%s"%(
                label,k,s,d, np.round((np.abs(b).max()/max(s,1e-300)),4)))
```

## `bcs_dimer.py`

Backs: **L1 SCF/Nambu** — CLEARED: Onsite s-wave anomalous prefactor in superscf.get_mf_anomalous / anomalous_term_ij_jit (the `2*v[d]` factor)

```python
"""Attractive-Hubbard dimer: pure-pairing (compute_normal=False) SCF vs the
analytic BCS gap equation.

Dimer, hopping t=1, spin-degenerate, mu=0 (half filling by particle-hole
symmetry).  Uniform onsite Delta.  BdG modes: eps_n = +-t, E_n=sqrt(eps^2+D^2).
<c_{i,dn} c_{i,up}> = -sum_n |phi_n(i)|^2 D/(2E_n) = -D/(2E)   (|phi|^2=1/2 each)
Gap equation Delta = U <c_dn c_up>  =>  D = -U D/(2E) = |U| D /(2E)
  => sqrt(t^2+D^2) = |U|/2  => D = sqrt(U^2/4 - t^2).
"""
import numpy as np, os
from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector

for f in ("MF.pkl",):
    if os.path.exists(f): os.remove(f)

t = 1.0
for U in (-3.0,-4.0,-5.0):
    g = geometry.dimer()
    h = g.get_hamiltonian()
    h.setup_nambu_spinor()
    h0 = h.copy()
    scf = meanfield.Vinteraction(h,U=U,mu=0.0,compute_normal=False,
            mix=0.3,maxerror=1e-8,maxite=500,load_mf=False,verbose=0)
    mf = np.array(scf.hamiltonian.intra)-np.array(h0.intra)
    eh = get_eh_sector(mf,i=0,j=1)
    ee = get_eh_sector(mf,i=0,j=0)
    D_code = np.abs(np.diag(eh))
    D_exact = np.sqrt(U**2/4.-t**2)
    print("U=%5.1f  converged=%s"%(U,getattr(scf,'converged',None)))
    print("   Delta(code) diag =",np.round(D_code,6))
    print("   Delta(BCS analytic) =",round(D_exact,6),
          "   ratio =",np.round(D_code/D_exact,6))
    print("   max|normal mf| =",np.max(np.abs(ee)))
```

## `bcs_engines.py`

Backs: **L1 SCF/Nambu** — CLEARED: Onsite s-wave anomalous prefactor in superscf.get_mf_anomalous / anomalous_term_ij_jit (the `2*v[d]` factor)

Backs: **L1 SCF/Nambu** — CLEARED: densitydensity.hubbard / hubbard_kpm building the onsite U matrix asymmetrically (v[2i,2i+1]=U with no v[2i+1,2i]) while Vinteraction uses the symmetric U/2 + U/2

```python
"""Same attractive-Hubbard dimer BCS oracle, through every public engine."""
import numpy as np, os
from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector
t=1.0 ; U=-4.0 ; D_exact = np.sqrt(U**2/4.-t**2)

def run(fn,**kw):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    g = geometry.dimer(); h = g.get_hamiltonian(); h.setup_nambu_spinor()
    h0 = h.copy()
    scf = fn(h,**kw)
    mf = np.array(scf.hamiltonian.intra)-np.array(h0.intra)
    return np.diag(get_eh_sector(mf,i=0,j=1)), np.max(np.abs(get_eh_sector(mf,i=0,j=0)))

common = dict(mu=0.0,compute_normal=False,mix=0.3,maxerror=1e-8,maxite=500,load_mf=False,verbose=0)
for name,fn,kw in [("Vinteraction", meanfield.Vinteraction, dict(U=U,**common)),
                   ("hubbardscf(=densitydensity.hubbard)", meanfield.hubbardscf, dict(U=U,**common)),
                   ("SzSz-style VJinteraction", meanfield.VJinteraction,
                       dict(U=U,mu=0.0,mix=0.3,maxerror=1e-8,maxite=500,verbose=0))]:
    try:
        D,nrm = run(fn,**kw)
        print("%-38s Delta=%s  ratio=%s  |normal mf|=%.2e"%(name,np.round(np.abs(D),6),
            np.round(np.abs(D)/D_exact,6),nrm))
    except Exception as e:
        print("%-38s FAILED %s"%(name,type(e).__name__,),e)
print("analytic BCS Delta =",D_exact)
```

## `bdg_band_energy.py`

```python
"""spectrum.total_energy sums eigenvalues below 0 with no Nambu awareness.
For a BdG Hamiltonian the electronic ground-state energy is
     E = (1/2) sum_{E_n<0} E_n^BdG + (1/2) Tr h
(H = (1/2) Psi^dag H_BdG Psi + (1/2) Tr h), so summing the BdG eigenvalues
gives 2E - Tr h instead.  With ZERO pairing this is checkable exactly:
     E_BdG_reported  ==  2*E_normal - Tr(h_intra)."""
import numpy as np
from pyqula import geometry
for name,gf in [("chain",geometry.chain),("honeycomb",geometry.honeycomb_lattice)]:
  for mu in (0.0,0.7):
    g = gf(); h = gf().get_hamiltonian(); h.shift_fermi(-mu)
    En = h.get_total_energy(nk=40)
    tr = np.trace(np.array(h.intra)).real
    hn = gf().get_hamiltonian(); hn.shift_fermi(-mu); hn.setup_nambu_spinor()
    Eb = hn.get_total_energy(nk=40)
    print("%-10s mu=%.1f  E_normal=%10.6f  Tr h=%8.4f  E_BdG(reported)=%10.6f   2E-Trh=%10.6f  diff=%.2e"%(
        name,mu,En,tr,Eb,2*En-tr,Eb-(2*En-tr)))
```

## `bug1_absolute_spatial_delta_sqrt2.py`

Backs: **L2 SC observables** — h.extract("absolute_spatial_delta") returns sqrt(2) times the true on-site gap on every spinful Nambu Hamiltonian

```python
"""BUG 1: h.extract("absolute_spatial_delta") returns sqrt(2)*|Delta_i|.

Oracle (three independent routes, all agreeing that |Delta_i| is the answer):
  * h.extract("swave")  -- reads the on-site anomalous matrix element directly
  * h.extract("absolute_delta") -- the sibling routine in the same file, which
    returns exactly sqrt(mean|Delta|^2)
  * the amplitude that was put in by add_swave

Cause: sctk/extract.py:245  `return np.sqrt(out.real/2.)`.  out is
full2profile(h, diag(m@m)) and htk/matrixcomponent.full2profile SUMS the FOUR
Nambu components of a spinful_nambu Hamiltonian (line 35), each contributing
|Delta_i|^2 -> 4|Delta_i|^2.  Dividing by 2 instead of 4 leaves 2|Delta_i|^2,
whose square root is sqrt(2)|Delta_i|.  The /2. is the divisor for the TWO
components of a spinless_nambu Hamiltonian -- but that Hilbert space cannot
reach this line at all (see bug2).
"""
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian()
h.add_swave(lambda r: 0.3 if r[0] < 0 else 0.1)   # 0.3 on A, 0.1 on B
true = np.abs(h.extract("swave"))
got  = h.extract("absolute_spatial_delta", nk=8)
print("true on-site |Delta_i|          :", np.round(true, 6))
print("absolute_spatial_delta          :", np.round(got, 6))
print("ratio                           :", np.round(got/true, 6), "   sqrt(2) =", round(np.sqrt(2), 6))
print("absolute_delta (sibling routine):", round(float(np.real(h.extract("absolute_delta", nk=8))), 6),
      "   sqrt(mean|D|^2) =", round(np.sqrt((0.3**2+0.1**2)/2), 6))
```

## `bug2_spinless_nambu_remove_nambu.py`

Backs: **L2 SC observables** — remove_nambu has no spinless_nambu branch, so every spinless BdG Hamiltonian is locked out of get_anomalous_hamiltonian and of the transport SC dispatch

```python
"""BUG 2: every spinless BdG (spinless_nambu) Hamiltonian is locked out of
get_anomalous_hamiltonian(), and with it of the SC extraction routines and of
the transport SC dispatch.

Cause: hamiltonians.Hamiltonian.remove_nambu (hamiltonians.py:623-632) branches on
spinful_nambu / spinful / spinless and has NO spinless_nambu branch, so a
spinless BdG falls through to `raise NotImplementedError("remove_nambu is not
implemented for this Hilbert space")`.  sctk/extract.get_anomalous_hamiltonian
(line 34) is `h0 = self.copy(); h0.remove_nambu(); h0.setup_nambu_spinor()`,
so it inherits the raise, and so does every caller of it.

The correct behaviour is obvious and already written for the spinful case:
strip the Nambu block, which for spinless_nambu is the 2x2 -> 1x1 (0,0) sector.
"""
import numpy as np, traceback
from pyqula import geometry, heterostructures

g = geometry.chain()
h = g.get_hamiltonian(has_spin=False)
h.add_swave(0.3)                      # the public way to build a spinless BdG
print("check_mode('spinless_nambu') :", h.check_mode("spinless_nambu"))
for label, f in [("h.remove_nambu()",              lambda: h.copy().remove_nambu()),
                 ("h.get_anomalous_hamiltonian()", lambda: h.get_anomalous_hamiltonian()),
                 ("h.extract('absolute_delta')",   lambda: h.extract("absolute_delta", nk=4)),
                 ("h.extract('deltak')",           lambda: h.extract("deltak", nk=4))]:
    try:
        f(); print(f"{label:32s} -> ok")
    except Exception as e: print(f"{label:32s} -> {type(e).__name__}: {str(e)[:70]}")

print("\n--- end to end: a spinless BdG junction ---")
h1 = g.get_hamiltonian(has_spin=False); h1.add_swave(0.2)
h2 = g.get_hamiltonian(has_spin=False); h2.add_swave(0.2)
ht = heterostructures.build(h1, h2)
try:
    print("ht.didv(energy=0.05) =", ht.didv(energy=0.05))
except Exception as e:
    print("ht.didv(energy=0.05) ->", type(e).__name__ + ":", str(e)[:80])
    print("   (raised inside transporttk/didv.py:89 _lead_is_superconducting"
          " -> h.get_anomalous_hamiltonian())")
```

## `bug3_sparse_superfluid_weight.py`

Backs: **L2 SC observables** — np.asarray on a sparse matrix makes all four superfluid-weight entry points crash on any is_sparse=True Hamiltonian

```python
"""BUG 3: every superfluid-weight entry point crashes on a sparse Hamiltonian.

Cause: sctk/superfluidweight.py:308  `hm.intra = np.asarray(hm.intra)`.
np.asarray on a scipy sparse matrix returns a 0-d OBJECT array (shape ()),
not a dense array, so the very next use -- electron_hole_signs' `n =
h.intra.shape[0]` (line 228) -- raises IndexError.  The same line also
mangles every stored hopping (`for t in hm.hopping: t.m = np.asarray(t.m)`).
algebra.todense() is what the rest of the package uses here.

TwistOperators is the constructor of all four public routes, so
h.get_superfluid_weight(), mode="finite_difference", decompose=True and
h.get_bkt_temperature() all fail identically.  is_sparse=True is a public
get_hamiltonian kwarg and add_swave propagates it.
"""
import numpy as np, traceback
from pyqula import geometry
g = geometry.square_lattice()
h = g.get_hamiltonian(is_sparse=True)
h.add_onsite(-0.6); h.add_swave(0.3)
print("h.is_sparse =", h.is_sparse, " type(h.intra) =", type(h.intra).__name__)
print("np.asarray(h.intra).shape =", np.asarray(h.intra).shape, " <- 0-d object array")
for label, f in [("h.get_superfluid_weight(nk=6)", lambda: h.get_superfluid_weight(nk=6)),
                 ("mode='finite_difference'", lambda: h.get_superfluid_weight(nk=6, mode="finite_difference")),
                 ("decompose=True", lambda: h.get_superfluid_weight(nk=6, decompose=True)),
                 ("h.get_bkt_temperature(nk=6)", lambda: h.get_bkt_temperature(nk=6))]:
    try: f(); print(f"{label:32s} -> ok")
    except Exception as e: print(f"{label:32s} -> {type(e).__name__}: {e}")
hd = g.get_hamiltonian(); hd.add_onsite(-0.6); hd.add_swave(0.3)
print("same model, dense            ->", np.round(hd.get_superfluid_weight(nk=6)[0,0], 6))
```

## `bug4_deltaud_mode.py`

Backs: **L2 SC observables** — Pairing mode "deltaud" raises NameError - the dispatch branch calls a function that does not exist

```python
"""BUG 4: pairing mode "deltaud" raises NameError.

Cause: sctk/pairing.py:22-23
      elif mode=="deltaud":
          weightf = lambda r1,r2: deltaud(r1,r2,deltaf)
There is no function named `deltaud` in the module (or anywhere in the
package); the one that exists is `get_deltaud` at sctk/pairing.py:230.
The lambda defers the lookup, so the branch is silently importable and only
blows up when a pairing is actually built.

This also defeats the self-diagnosing-error convention: pairing_generator's
else-branch tells the user the mode must be "one of the modes listed in
sctk.pairing.pairing_generator", and "deltaud" IS listed there.
"""
from pyqula import geometry
h = geometry.square_lattice().get_hamiltonian()
try:
    h.add_pairing(delta=0.2, mode="deltaud")
    print("built ok")
except Exception as e:
    print("h.add_pairing(delta=0.2, mode='deltaud') ->", type(e).__name__ + ":", e)
import pyqula.sctk.pairing as P
print("hasattr(pairing,'deltaud')     =", hasattr(P, "deltaud"))
print("hasattr(pairing,'get_deltaud') =", hasattr(P, "get_deltaud"))
```

## `bug5_circular_import.py`

Backs: **L2 SC observables** — pyqula.sctk.extract and pyqula.sctk.dvector cannot be the first pyqula module imported (circular import with superconductivity.py)

```python
"""BUG 5: pyqula.sctk.extract and pyqula.sctk.dvector cannot be the first
pyqula module imported -- circular import with superconductivity.py.

sctk/extract.py:2 does `from ..superconductivity import get_eh_sector`, while
superconductivity.py:341 does `from .sctk.extract import extract_pairing`
(and :293 `from .sctk.dvector import dvector2deltas`).  Whichever of the two
is imported first works; importing the sctk module first leaves
superconductivity.py executing against a half-initialised sctk.extract.
"""
import subprocess, sys, os
env = dict(os.environ)
for stmt in ["from pyqula.sctk.extract import extract_triplet_pairing",
             "from pyqula.sctk.dvector import matrix2dvector",
             "import pyqula.superconductivity; from pyqula.sctk.extract import extract_triplet_pairing",
             "from pyqula.sctk.superfluidweight import superfluid_weight"]:
    r = subprocess.run([sys.executable, "-c", stmt + "; print('OK')"],
                       capture_output=True, text=True, env=env)
    last = (r.stdout.strip() or r.stderr.strip().splitlines()[-1])
    print(f"{stmt[:62]:64s} -> {last[:70]}")
```

## `bug6_singlet_operator.py`

Backs: **L2 SC observables** — The "singlet" pairing operator has no Hilbert-space guard, while the four other pairing operators in the same registry do

```python
"""HOLE: the "singlet" pairing operator has no Hilbert-space guard, while the
four other pairing operators in the same registry do.

operatorlist.py:45-48 route "spair"/"deltax"/"deltay"/"deltaz" to
operators.get_pairing, which raises ValueError unless h.has_eh (operators.py:228)
and NotImplementedError unless spinful_nambu (:231).  operatorlist.py:73 routes
"singlet" to sctk/operator.py:real_singlet, which has no guard at all:
    op = h.copy()*0.
    op.add_swave(1.0)      # <- PROMOTES op into Nambu space
    return Operator(op.intra)
On a non-Nambu h the returned operator is twice the Hilbert-space dimension,
and the mismatch only surfaces when it meets an eigenvector, as a raw numpy
matmul message naming neither pyqula nor the requirement.
"""
import numpy as np
from pyqula import geometry
g = geometry.chain()
cases = [("spinful non-Nambu",  lambda: g.get_hamiltonian()),
         ("spinless non-Nambu", lambda: g.get_hamiltonian(has_spin=False)),
         ("spinless_nambu",     lambda: (lambda h: (h.add_swave(0.2), h)[1])(g.get_hamiltonian(has_spin=False)))]
for lbl, mk in cases:
    h = mk()
    for n in ["spair", "singlet"]:
        try:
            m = h.get_operator(n).get_matrix()
            print(f"{lbl:19s} {n:8s} -> built shape {m.shape}, h.intra {h.intra.shape}")
        except Exception as e:
            print(f"{lbl:19s} {n:8s} -> {type(e).__name__}: {str(e)[:55]}")
print()
h = geometry.square_lattice().get_hamiltonian()   # spinful, non-Nambu
try: h.get_bands(operator=h.get_operator("singlet"), nk=3)
except Exception as e: print("applying the 4x4 'singlet' to a 2x2 h ->", type(e).__name__+":", str(e)[:95])
```

## `bug7_identify_sc_spinless_nambu.py`

Backs: **L2 SC observables** — identify_superconductivity has no spinless_nambu branch - a sibling of the bug_audit 4.2 d-vector guard that the fix did not update

```python
"""HOLE (sibling of the bug_audit 4.2 d-vector guard, 05e0f17):
superconductivity.identify_superconductivity has no spinless_nambu branch.

superconductivity.py:442 guards only `if not h.has_eh: return []`, then
line 447 calls `h.get_average_dvector()` unconditionally.  05e0f17 gave the
d-vector a spinful_nambu guard (correctly), but this caller -- the one place
that asks for the d-vector without knowing the Hilbert space -- was not given
the matching branch, so on a spinless BdG the symmetry identifier now raises
a message about the d-vector instead of reporting the pairing it can see.
(dict2absdeltas at :471 has the same 4-per-site assumption.)

Reachable through meanfield.identify_symmetry_breaking (meanfield.py:509),
which is the public function behind SCF.identify_symmetry_breaking
(scftk/densitydensity.py:727) and is used in three examples/.
"""
import numpy as np
from pyqula import geometry, meanfield
g = geometry.chain()
h0 = g.get_hamiltonian(has_spin=False); h0.setup_nambu_spinor()
h  = g.get_hamiltonian(has_spin=False); h.add_swave(0.3)
print("spinless_nambu, |Delta| = 0.3")
try:
    print("identify_symmetry_breaking ->", meanfield.identify_symmetry_breaking(h0, h))
except Exception as e:
    print("identify_symmetry_breaking ->", type(e).__name__ + ":", str(e)[:95])
hs0 = g.get_hamiltonian(); hs0.setup_nambu_spinor()
hs  = g.get_hamiltonian(); hs.add_swave(0.3)
print("same model, spinful_nambu  ->", meanfield.identify_symmetry_breaking(hs0, hs))
```

## `calib_swave.py`

```python
import numpy as np
from pyqula import geometry
from pyqula.superconductivity import get_eh_sector

g = geometry.dimer()
h = g.get_hamiltonian()
print("hopping matrix (spinful):\n", np.round(np.array(h.intra),3))
h0 = h.copy()
h.setup_nambu_spinor()
hn0 = h.copy()
D = 0.3
h.add_swave(D)
mf = np.array(h.intra) - np.array(hn0.intra)
print("nambu intra shape", h.intra.shape)
print("eh sector (0,1) of add_swave(%g):\n"%D, np.round(get_eh_sector(mf,i=0,j=1),3))
print("eh sector (0,0):\n", np.round(get_eh_sector(mf,i=0,j=0),3))
(k,e) = h.get_bands()
print("spectrum:", np.round(np.sort(np.array(e)),4))
```

## `cleared_phs_all_pairing_modes.py`

Backs: **L2 SC observables** — CLEARED: Particle-hole symmetry and Hermiticity of all 23 working pairing modes

```python
import numpy as np, warnings
warnings.filterwarnings("ignore")
from pyqula import geometry
modes = ["swave","extended_swave","triplet","pwave","nodal_fwave","chiral_pwave",
         "chiral_fwave","chiral_dwave","chiral_gwave","antihaldane","haldane",
         "swavez","px","dpid","swaveA","swaveB","swavesublattice","dx2y2",
         "nodal_dwave","dxy","snn","C3nn","SnnAB","deltaud"]
g = geometry.honeycomb_lattice()
k = np.array([0.137,0.291,0.])
for m in modes:
    try:
        h = g.get_hamiltonian()
        h.add_pairing(delta=0.3, mode=m, d=[0.,0.,1.])
    except Exception as e:
        print(f"{m:18s} BUILD-FAIL {type(e).__name__}: {str(e)[:60]}"); continue
    try:
        hk = h.get_hk_gen()
        m1 = np.array(hk(k)); m2 = np.array(hk(-k))
        herm = np.max(np.abs(m1-np.conjugate(m1.T)))
        e1 = np.sort(np.linalg.eigvalsh(m1)); e2 = np.sort(np.linalg.eigvalsh(m2))
        phs = np.max(np.abs(e1 + e2[::-1]))   # E_n(k) = -E_n(-k)
        anom = np.max(np.abs(m1)) # nonzero check
        from pyqula.superconductivity import get_eh_sector
        a01 = np.max(np.abs(get_eh_sector(m1,i=0,j=1)))
        print(f"{m:18s} herm={herm:.2e}  PHS_resid={phs:.2e}  |anom|={a01:.4f}")
    except Exception as e:
        print(f"{m:18s} EVAL-FAIL {type(e).__name__}: {str(e)[:70]}")
```

## `constrain2.py`

```python
import numpy as np, os
from pyqula import geometry, meanfield
g = geometry.chain()
for con in ([],["no_magnetism"]):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = g.get_hamiltonian()
    s = meanfield.Vinteraction(h,V1=3.0,filling=0.3,nk=40,mf="ferroZ",
            constrains=con,mix=0.3,maxerror=1e-8,maxite=2000,load_mf=False,verbose=0)
    print("constrains:",con,"converged",s.converged)
    for k in sorted(s.mf,key=str):
        print("  mf[%s]=\n"%(k,),np.round(np.array(s.mf[k]),6))
    hc=s.hamiltonian
    print("  magnetization:",np.round(hc.get_magnetization(nk=40),6))
    print("  get_vev mz:",np.round(hc.get_vev("mz",nk=40),6),
          " mx:",np.round(hc.get_vev("mx",nk=40),6))
```

## `constrain_bonds.py`

```python
"""mfconstrains' remove_* functions only ever touch mf[(0,0,0)].  With an
intersite interaction the Fock (exchange) mean field lives on the BONDS and is
spin dependent, so 'no_magnetism' need not actually leave a non-magnetic state."""
import numpy as np, os
from pyqula import geometry, meanfield

g = geometry.chain()
for V1,filling in [(3.0,0.3),(4.0,0.25),(3.0,0.5)]:
  for con in ([],["no_magnetism"]):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = g.get_hamiltonian()
    s = meanfield.Vinteraction(h,V1=V1,filling=filling,nk=40,mf="ferroZ",
            constrains=con,mix=0.3,maxerror=1e-8,maxite=2000,load_mf=False,verbose=0)
    hc = s.hamiltonian
    mx = hc.get_vev("mx",nk=40); my = hc.get_vev("my",nk=40); mz = hc.get_vev("mz",nk=40)
    mf000 = np.array(s.mf[(0,0,0)])
    onsite_mz = ((mf000[0,0]-mf000[1,1])/2).real
    bond_keys=[k for k in s.mf if k!=(0,0,0)]
    bond_spin = 0.
    for k in bond_keys:
        m=np.array(s.mf[k]); bond_spin=max(bond_spin,np.max(np.abs(m[0::2,0::2]-m[1::2,1::2])))
    print("V1=%.1f fill=%.2f con=%-16s : <mz>=%+.6f  onsite mf mz=%+.2e  max|mf_up-mf_dn| on bonds=%.3e"%(
        V1,filling,str(con),mz[0],onsite_mz,bond_spin))
```

## `defaults.py`

```python
import re,sys,inspect,os
sys.path.insert(0,"<repo root>/src")
R="<repo root>/"
lines=open(R+"documentation/user_guide.md").read().split("\n")
for i,l in enumerate(lines):
    if l.strip()=="# Main functions and methods": start=i;break
from pyqula import geometry
g=geometry.honeycomb_lattice()
h=g.get_hamiltonian()
objs={"h":h,"g":g}
cur=None;rep=[]
for i in range(start,len(lines)):
    l=lines[i]
    s=l.strip()
    m=re.match(r"^### (h|g)\.([A-Za-z_0-9]+)\(\)\s*$",s)
    if m: cur=(m.group(1),m.group(2),i+1); continue
    if s.startswith("### "): cur=None; continue
    if cur is None: continue
    m2=re.match(r"^-\s*`?([A-Za-z_][A-Za-z_0-9]*)`?\s*=\s*([^\s:,]+)\s*[:,]",s)
    if not m2: continue
    arg,docval=m2.group(1),m2.group(2).strip("`")
    pre,meth,ln=cur
    ob=objs[pre]
    f=getattr(ob,meth,None)
    if f is None: continue
    try: sig=inspect.signature(f)
    except Exception: continue
    if arg not in sig.parameters:
        rep.append((ln,i+1,"%s.%s"%(pre,meth),arg,docval,"NOT IN SIGNATURE (kwargs-only?)"))
        continue
    d=sig.parameters[arg].default
    if d is inspect._empty: rep.append((ln,i+1,"%s.%s"%(pre,meth),arg,docval,"no default (required)")); continue
    ds=str(d)
    if ds.replace(".0","")!=docval.replace(".0","") and docval not in (ds,repr(d)):
        rep.append((ln,i+1,"%s.%s"%(pre,meth),arg,docval,"code default = "+ds))
for r in rep: print(r)
print("suspects:",len(rep))
```

## `e1.py`

```python
from edit import apply
pairs = []

# --- errors section: the exemplar message, and the new-raise class
pairs.append((
"""ValueError: unknown mode ED2; the DOS accepts 'ED', 'KPM' and 'adaptive'
```""",
"""ValueError: unknown mode ED2; the DOS accepts 'ED', 'KPM', 'adaptive', 'Green' and 'RG'
```"""))

pairs.append((
"""Two of these messages are worth knowing in advance because they point at the fix
rather than the failure: a superconducting quantity asked of a normal Hamiltonian says
to call `h.setup_nambu_spinor()` first, and a spin quantity asked of a spinless one
says to call `h.turn_spinful()`.
""",
"""Two of these messages are worth knowing in advance because they point at the fix
rather than the failure: a superconducting quantity asked of a normal Hamiltonian says
to call `h.setup_nambu_spinor()` first, and a spin quantity asked of a spinless one
says to call `h.turn_spinful()`.

A second family of checks exists not because the routine cannot run, but because it
could run and return a number that means nothing. Those raise rather than answer:

- `h.get_average_spin_splitting()`, `h.get_spin_splitting_density()` and
  `h.get_spin_splitting_vs_energy()` refuse a Hamiltonian whose spin off-diagonal
  block does not vanish (Rashba, any spin-orbit term, non-collinear order), naming
  the largest off-diagonal element found -- they are built on `remove_spin`, which
  would drop that block without warning
- `h.get_dos()` on a non-Hermitian Hamiltonian refuses every mode but `"ED"`, and
  refuses `use_kpm=True`: the Chebyshev and adaptive expansions assume a real
  spectrum (see "Non-Hermitian Hamiltonians")
- `h.get_kdos_bands(frand=...)` refuses unless `mode="KPM"`, since only the KPM
  path draws the random vectors that argument supplies
- `h.get_total_energy(fermi=...)` refuses a Nambu Hamiltonian, where shifting the
  occupation cut does not shift the electronic energy -- use `h.shift_fermi()` on
  the Hamiltonian instead
- `h.get_multildos(projection="atomic",operator=...)` refuses the combination and
  names `projection="TB"` as the one that projects
- the surface density of states refuses a momentum-dependent operator such as
  `"valley"`, which its Green's function has already integrated over
- the lead decimation behind `HT.didv()`, `h.get_dos(mode="RG")` and
  `Embedding.get_gf()` refuses an energy sitting exactly on a lead level with a
  broadening below about `1e-7`, naming the energy, the broadening, the residual it
  reached and the tolerance it needed, and recommending a larger `delta`
- `h.check()` raises on a Nambu Hamiltonian whose electron-hole symmetry is broken,
  naming the deviation, instead of printing a line and terminating the interpreter

Passing an argument a routine cannot honour is also an error rather than a silent
drop: an unknown keyword to `h.get_multildos()` is a `TypeError`, and so is passing
both the old `op=` and the new `operator=` spelling to it.
"""))

apply(pairs)
```

## `e10.py`

```python
from edit import apply
pairs = []

pairs.append((
"""### g.get_hamiltonian()
Generate the Hamiltonian from a geometry.

Optional arguments

- tij = [1.0,.0,0.]: List with 1st, 2nd, 3rd nearest neighbor hopping

Returns the Hamiltonian
""",
"""### g.get_hamiltonian()
Generate the Hamiltonian from a geometry.

Optional arguments

- tij = [1.0,.0,0.]: List with 1st, 2nd, 3rd nearest neighbor hopping, or a
  function of two positions returning the hopping between them, or a
  `specialhopping.HoppingGenerator`
- has_spin=True: include the spin degree of freedom
- is_sparse=False: store the matrices in sparse form, for large cells
- non_hermitian=False: build a non-Hermitian Hamiltonian, which routes the
  band structure, DOS, LDOS and Berry curvature to their non-Hermitian
  implementations (see "Non-Hermitian Hamiltonians")
- is_multicell=False: store the hoppings as a multicell dictionary
- nc=2: neighbor cutoff used by the multicell construction

An unrecognized keyword raises `TypeError` listing the accepted ones, so a
misspelled `has_spin` is not silently the default.

Returns the Hamiltonian

### g.remove()
Return a copy of the geometry with the listed sites removed, indexed into
`g.r`. Used to carve a vacancy or an antidot out of a flake or a supercell
before building its Hamiltonian (see "Valley operator" for an example).
Takes the list of site indices to drop, and returns a new geometry --
`g` itself is untouched
"""))

# --- h.get_bands(): eigmode / non-Hermitian
pairs.append((
"""Without `kpath` the path is $\\Gamma$-M for a square-like 2D lattice,""",
"""- eigmode="complex": non-Hermitian Hamiltonians only -- which part of the
  complex eigenvalue is returned and written, `"complex"`, `"real"` or
  `"imag"` (see "Non-Hermitian Hamiltonians"). With `"complex"` the written
  `BANDS.OUT` carries `k`, `Re E`, `Im E` and then the operator columns

Without `kpath` the path is $\\Gamma$-M for a square-like 2D lattice,"""))

# --- h.get_ldos(): eigmode
pairs.append((
"""- projection="TB": `"TB"`, `"TBRS"` (real-space interpolated) or `"atomic"`

Return x, position, y position and LDOS""",
"""- projection="TB": `"TB"`, `"TBRS"` (real-space interpolated) or `"atomic"`

- eigmode="complex": non-Hermitian Hamiltonians only -- whether `e` is read
  on the real or the imaginary axis of the complex spectrum (see
  "Non-Hermitian Hamiltonians"). Only `mode="diagonalization"` is
  implemented there

Return x, position, y position and LDOS"""))

apply(pairs)
```

## `e11.py`

```python
from edit import apply
apply([(
"""### g.remove()
Return a copy of the geometry with the listed sites removed, indexed into
`g.r`. Used to carve a vacancy or an antidot out of a flake or a supercell
before building its Hamiltonian (see "Valley operator" for an example).
Takes the list of site indices to drop, and returns a new geometry --
`g` itself is untouched
""",
"""### g.remove()
Return a copy of the geometry with sites removed. Used to carve a vacancy
or an antidot out of a flake or a supercell before building its
Hamiltonian (see "Valley operator" for an example).

Arguments

- i=0: which sites to drop -- a single index into `g.r`, a list of indices,
  or a callable of the position, in which case every site where it returns
  True is removed

Returns a new geometry; `g` itself is untouched
""")])
```

## `e12.py`

```python
from edit import apply

NEW = r"""### h.add_haldane()
Add a Haldane term: a complex second-neighbor hopping whose sign is set by
the chirality of the two-step path, which opens a gap and makes a
honeycomb lattice a Chern insulator (see "Chern number").

Arguments

- t: amplitude of the second-neighbor hopping, a number or a callable of the position

Needs a geometry with a sublattice. Breaks time-reversal symmetry

### h.add_modified_haldane() / h.add_antihaldane()
Same second-neighbor complex hopping, but with the sign also flipped
between the two sublattices, so the two valleys acquire opposite masses
and the total Chern number is zero -- a valley-Hall rather than a Chern
insulator. `add_antihaldane` is a second name for the same method.

Arguments

- t: amplitude of the second-neighbor hopping

### h.add_kane_mele()
Add a Kane-Mele spin-orbit term: the Haldane hopping with opposite sign for
the two spin channels, so time-reversal symmetry is preserved and the model
is a quantum spin Hall insulator instead of a Chern insulator. Turns the
Hamiltonian spinful if it is not already.

Arguments

- t: amplitude of the spin-orbit second-neighbor hopping

Its spin Chern number is what `topology.spin_chern` computes

### h.add_anti_kane_mele()
The sublattice-staggered counterpart of `add_kane_mele`, in the same way
`add_modified_haldane` is the counterpart of `add_haldane`.

Arguments

- t: amplitude of the second-neighbor hopping

### h.add_kekule() / h.add_chiral_kekule()
Add a Kekule bond modulation: a period-tripling pattern on the
first-neighbor bonds of a honeycomb lattice, which folds the two Dirac
points onto $\Gamma$ and gaps them. `add_kekule(t)` takes the modulation
amplitude. `add_chiral_kekule(t1=...,t2=...)` adds the bond-direction-aware
chiral version, whose two amplitudes are the two independent complex
components of the modulation, optionally on an explicit `registry=` of
retained hexagon centers rather than the default one

### h.add_valley_exchange()
Add a valley-space exchange term
$\vec{v}\cdot(\tau_x,\tau_y,\tau_z)$, the valley-pseudospin analogue of
`add_exchange` for real spin (see "In-plane valley operators").

Arguments

- v = (vx,vy,vz): the valley field

### h.add_peierls() / h.add_orbital_magnetic_field()
Add an out-of-plane orbital magnetic field as a Peierls phase on every
hopping. `add_orbital_magnetic_field` is a second name for the same method.

Arguments

- mag_field: the field, in flux quanta per unit cell of the lattice. For a
  commensurate calculation this must be chosen so that the flux through the
  cell is a rational multiple of the flux quantum, i.e. build the supercell
  first and pick the field to match it

Optional arguments

- gauge="Landau": `"Landau"` or `"symmetric"`

Refuses a Hamiltonian that already carries a pairing amplitude
(`NotImplementedError`): a Cooper pair has charge 2e, so the anomalous term
has no single Peierls phase, and an orbital field in a superconductor means
vortices. Add the field to the normal-state Hamiltonian first, then
`h.turn_nambu()`/`h.add_swave()`

### h.add_inplane_bfield()
Add an in-plane magnetic field, as a Peierls phase built from the
out-of-plane coordinate -- meaningful for a multilayer or a system with
finite thickness, where an in-plane field still threads flux between the
layers.

Optional arguments

- b=0.0: field strength
- phi=0.0: in-plane direction of the field, in units of $\pi$

Only implemented up to two dimensions

### h.add_strain()
Modify the hoppings according to a strain field, the standard way of
producing a pseudo-magnetic field in graphene.

Arguments

- sr: a callable of the position returning the local strain

Optional arguments

- mode="scalar": `"scalar"` rescales every hopping by `sr(r)` alone;
  `"directional"` also uses the bond direction, for a genuinely
  anisotropic strain

Turns the Hamiltonian multicell. Not implemented for a spinless Nambu
Hilbert space

### h.add_crystal_field()
Add a crystal field: an onsite potential built from the local atomic
environment, so that sites with fewer or more distant neighbors (an edge,
a vacancy, the two inequivalent stackings of a bilayer) sit at different
energies.

Arguments

- v: strength of the crystal field. The built-in potential is normalized
  and has its average removed, so `v` sets the spread of the onsite
  energies rather than their offset

Optional arguments

- rcut=6.0: distance cutoff of the neighbor sum

### h.generate_spin_spiral()
Rotate the Hamiltonian into a spin-spiral ansatz, i.e. impose a magnetic
texture whose quantization axis winds with a wavevector $q$. Used both as a
starting point for a spiral mean-field calculation and to scan the energy
of the spiral as a function of $q$ (see "Spin-spin exchange interactions").

Optional arguments

- vector=[0.,0.,1.]: axis the spins rotate about
- qspiral=[1.,0.,0.]: the spiral wavevector
- fractional=True: read `qspiral` in fractional (reduced) coordinates. With
  `False` only the inter-cell hoppings are rotated

Needs a spinful Hamiltonian

### h.add_pairing()
Add a general superconducting pairing term, given by its symmetry channel,
turning the Hamiltonian into its Nambu (BdG) form if it is not already (see
"Spin-triplet d-vector and non-unitary superconductivity").

Optional arguments

- delta=0.0: pairing amplitude, a number or a callable of the position
- mode="swave": the pairing symmetry. The accepted names are listed in
  `pyqula.sctk.pairing.pairing_modes` -- `"swave"`, `"extended_swave"`,
  `"triplet"`, `"pwave"`, `"chiral_pwave"`, `"dx2y2"`, `"dxy"`,
  `"nodal_dwave"`, `"chiral_dwave"`, `"nodal_fwave"`, `"chiral_fwave"`,
  `"chiral_gwave"`, `"haldane"`, `"antihaldane"`, and several others. A
  callable returning the 2x2 pairing matrix for a pair of positions is also
  accepted. An unknown name raises `ValueError` listing every accepted one
- d=[0.,0.,1.]: the d-vector, for the triplet channels

### h.setup_nambu_spinor()
Put the Hamiltonian into its Nambu (electron-hole doubled) form, with zero
pairing. This is the call the error messages of the superconducting
routines point at: a superconducting quantity asked of a normal Hamiltonian
needs the electron-hole degree of freedom to exist first. Equivalent to
`h.add_swave(0.0)`. Modifies in place

### h.turn_nambu()
Add the electron-hole degree of freedom without adding any pairing. Lower
level than `setup_nambu_spinor()`, which is what user code normally wants.
Modifies in place

### h.turn_spinful()
Add the spin degree of freedom to a spinless Hamiltonian, doubling every
matrix with an identity in spin space. This is the call the error messages
of the spin-resolved routines point at. Modifies in place, and does nothing
if the Hamiltonian is already spinful

### h.remove_spin()
The inverse: keep one spin block and discard the other, returning to a
spinless Hamiltonian.

Optional arguments

- channel="up": which block to keep, `"up"` or `"dn"`

This **drops the spin off-diagonal block**, so it is only meaningful when
spin is a good quantum number -- with spin-orbit coupling or non-collinear
order the result is a different model, not a projection of this one. The
spin-splitting routines built on it check that explicitly (see
"Errors and unsupported inputs")

### h.turn_multicell()
Convert the Hamiltonian to multicell form, i.e. store the inter-cell
hoppings as a dictionary keyed by lattice vector instead of the single
`inter` matrix of the nearest-neighbor-cell form. Several routines require
it, and their error messages say so; hoppings beyond the first neighboring
cell need it. Modifies in place. `h.get_no_multicell()` returns a
non-multicell copy where that is possible

### h.get_hopping_dict()
Return the real-space hoppings as a dictionary `{(n1,n2,n3): matrix}`,
keyed by the lattice vector connecting the two cells, with `(0,0,0)` the
intracell block. This is the representation the response functions accept
for an interaction with support beyond one cell (see "Interactions beyond
onsite"), and the natural way to inspect or modify a Hamiltonian's hoppings
directly. `h.get_multihopping()` returns the same thing wrapped in a
`multihopping.MultiHopping`, which supports addition and scalar
multiplication, and `h.set_multihopping()` writes one back

### h.get_rkky()
Compute the RKKY interaction between two magnetic impurities mediated by
the electrons of `h`, i.e. the effective exchange coupling as a function of
their separation.

Optional arguments

- mode="pm": `"pm"` computes it explicitly ("poor man's"), by adding two
  local exchange fields to the Hamiltonian and taking the energy difference
  between their parallel and antiparallel alignment; `"LR"` computes it from
  linear response instead, which is much faster for a map over many
  separations. An unknown mode raises `ValueError` listing both
- for `mode="pm"`: `ri`, `rj` the positions of the two impurities (both
  required), `nk=10`, `dj=1e-1` the strength of the probe exchange fields
- for `mode="LR"`: `R=[0,0,0]` the lattice vector between the two cells,
  `ii=0`, `jj=0` the sites inside them, `nk=100`, `delta` the analytic
  continuation (default `1/nk`)

`rkky.rkky_map(h,n=...)` sweeps either mode over a range of separations.
On a bipartite lattice at half filling the sign follows the usual theorem:
ferromagnetic between sites of the same sublattice, antiferromagnetic
between opposite ones

### h.get_ldos()"""

apply([("### h.get_ldos()",NEW)])
```

## `e13.py`

```python
from edit import apply
apply([(
"""- mag_field: the field, in flux quanta per unit cell of the lattice. For a
  commensurate calculation this must be chosen so that the flux through the
  cell is a rational multiple of the flux quantum, i.e. build the supercell
  first and pick the field to match it""",
"""- mag_field: the field, in units of the flux quantum per unit area of the
  lattice -- the Peierls phase on a bond is $2\\pi B\\,y\\,dx$ in the Landau
  gauge. A periodic (commensurate) calculation needs the flux through the
  unit cell to be a rational multiple of the flux quantum, so build the
  supercell first and pick `mag_field` to match it""")])
```

## `e14.py`

```python
from edit import apply
apply([(
"""### h.add_valley_exchange()
Add a valley-space exchange term
$\\vec{v}\\cdot(\\tau_x,\\tau_y,\\tau_z)$, the valley-pseudospin analogue of
`add_exchange` for real spin (see "In-plane valley operators").

Arguments

- v = (vx,vy,vz): the valley field
""",
"""### h.add_valley_exchange()
Add a valley-space exchange term
$\\vec{v}\\cdot(\\tau_x,\\tau_y,\\tau_z)$, the valley-pseudospin analogue of
`add_exchange` for real spin (see "In-plane valley operators").

Arguments

- v = (vx,vy,vz): the valley field

Built from the same in-plane valley operators, so for a periodic
Hamiltonian it needs a Kekule-commensurate cell -- a multiple-of-3
supercell of the primitive honeycomb cell -- and raises `ValueError`
otherwise. A finite (0d) flake needs no such commensurability
""")])
```

## `e15.py`

```python
from edit import apply
NEW = r"""### h.get_hk_gen()
Return the Bloch generator: a function of the reduced k-vector returning
the Bloch matrix $H(k)$ of the Hamiltonian. This is the object every
k-space routine in the library is built on, and the way to evaluate a
model at a chosen k-point directly (`h.get_hk_gen()([0.3,0.,0.])`).
pyqula's convention is the periodic one, $H(k)=\sum_R t(R)e^{2\pi i k\cdot R}$
with $k$ in reduced coordinates and no intra-cell atomic positions in the
phases

### h.get_gf()
Return the bulk Green's function of the Hamiltonian at one energy, as a
matrix in the same basis as `h.intra`, obtained by integrating over the
Brillouin zone.

Optional arguments

- energy=0.0: energy at which it is evaluated
- delta=1e-5: imaginary part (analytic continuation)
- mode="adaptive": how the Brillouin-zone integral is done -- `"adaptive"`
  (error-controlled), `"full"` (a fixed `nk` mesh) or `"renormalization"`
- gtype="bulk": `"bulk"` or `"surface"`

`-Im Tr G/\pi` is the density of states, which is what
`h.get_dos(mode="Green")` computes from it

### h.get_hopping_dict()"""
apply([("### h.get_hopping_dict()",NEW)])
```

## `e16.py`

```python
from edit import apply
pairs = []
pairs.append((
"""- operator=None: operator used to weight the spectral function, e.g. `"unfold"` (see "Electronic structure folding and unfolding")

- energies, delta, nk: frequency range, broadening, k-point density

Returns k-path fraction, energy and spectral weight""",
"""- operator=None: operator used to weight the spectral function, e.g. `"unfold"` (see "Electronic structure folding and unfolding")

- energies, delta, nk: frequency range, broadening, k-point density

- mode="ED": `"ED"` or `"KPM"` (equivalently `use_kpm=True`)

- frand=None: generator of the random vectors the KPM stochastic trace
  draws. Only the KPM path uses them, so passing it without `mode="KPM"`
  raises `ValueError` saying so, rather than being dropped

Returns k-path fraction, energy and spectral weight"""))

pairs.append((
"""`get_ldos` returns the real-space positions and the LDOS profile in a window of `nsuper` unit cells around the defect, showing e.g. Friedel oscillations or bound/in-gap states induced by the impurity (it also writes `LDOS.OUT`; pass `write=False` to suppress that). `eb.get_dos()` gives the total DOS, `eb.multildos()` scans the LDOS over many energies (written to a `MULTILDOS/` folder), and `eb.get_didv()` computes transport through the embedded defect.""",
"""`get_ldos` returns the real-space positions and the LDOS profile in a window of `nsuper` unit cells around the defect, showing e.g. Friedel oscillations or bound/in-gap states induced by the impurity (it also writes `LDOS.OUT`; pass `write=False` to suppress that). `eb.get_dos()` gives the total DOS, `eb.multildos()` scans the LDOS over many energies (written to a `MULTILDOS/` folder), `eb.get_gf()` returns the embedded Green's function itself (with `operator=` it returns $AG$, the same convention `get_ldos` uses), and `eb.get_didv()` computes transport through the embedded defect."""))
apply(pairs)
```

## `e17.py`

```python
from edit import apply
apply([(
"""`ds` and `db` are, respectively, the surface and bulk spectral weight at each `(k,e)`; plotting `k,e` colored by `ds` shows the topologically-protected edge states living at the boundary, absent from the bulk spectrum `db`. See `examples/readme_examples/surface_2dTI/main.py` for a runnable version.
""",
"""`ds` and `db` are, respectively, the surface and bulk spectral weight at each `(k,e)`; plotting `k,e` colored by `ds` shows the topologically-protected edge states living at the boundary, absent from the bulk spectrum `db`. See `examples/readme_examples/surface_2dTI/main.py` for a runnable version.

The k-integrated counterpart is `dos.surface_dos(h,...)`, which returns the
surface and bulk densities of states at a set of energies. It takes an
`operator=` and projects onto it, so e.g. the spin-resolved surface DOS of
the model above separates the two counter-propagating edge channels. A
momentum-dependent operator such as `"valley"` is refused with a
`NotImplementedError` naming it, since the Green's function it is built on
has already been integrated over the Brillouin zone.

This whole family -- `dos.surface_dos`, `dos.dos_surface`,
`dos.bulkandsurface`, `dos.surface2bulk` and the surface writers in
`kdos` -- reports $-\\mathrm{Im}\\,\\mathrm{Tr}\\,G$ without the $1/\\pi$ that
`h.get_dos()` applies, so their values are $\\pi$ times a density of states.
They agree with each other; compare them among themselves rather than
against `h.get_dos()`.
""")])
```

## `e18.py`

```python
from edit import apply
apply([(
"""`operator=` and projects onto it, so e.g. the spin-resolved surface DOS of
the model above separates the two counter-propagating edge channels. A
momentum-dependent operator such as `"valley"` is refused with a""",
"""`operator=` and projects onto it, giving e.g. the spin-resolved surface DOS
of a magnetized lead rather than the charge one. A
momentum-dependent operator such as `"valley"` is refused with a""")])
```

## `e19.py`

```python
from edit import apply
apply([(
"""```python
imax = np.argmax(es.imag) # the most amplified state
(x,y,d) = h.get_ldos(e=es[imax].imag,delta=1e-2,eigmode="imag",nrep=1)
print("it lives at x =",x[np.argmax(d)])
```""",
"""```python
import numpy as np
from pyqula import geometry
n = 20
g = geometry.chain().get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))
(ks,es) = h.get_bands(kpath=[[0.,0.,0.]],write=False)

imax = np.argmax(es.imag) # the most amplified state
(x,y,d) = h.get_ldos(e=es[imax].imag,delta=1e-2,eigmode="imag",nrep=1)
print("it lives at x =",x[np.argmax(d)])
```""")])
```

## `e2.py`

```python
from edit import apply
pairs = []

# --- DOS section: mode list and nk
pairs.append((
"""- mode: how the DOS is computed -- `"ED"` (default, broadens a k-mesh band structure), `"Green"` (sums a Green's function per energy, useful when only a handful of energies are needed), `"KPM"` (Chebyshev kernel-polynomial expansion, for large sparse systems -- see the "Chebyshev kernel polynomial (KPM) methods" section), or `"adaptive"`
- nk: number of k-points in the mesh (`"ED"`/`"KPM"` modes)""",
"""- mode: how the DOS is computed -- `"ED"` (default, broadens a k-mesh band structure), `"Green"`/`"RG"` (sums a Green's function per energy, useful when only a handful of energies are needed), `"KPM"` (Chebyshev kernel-polynomial expansion, for large sparse systems -- see the "Chebyshev kernel polynomial (KPM) methods" section), or `"adaptive"`
- nk: number of k-points in the mesh. `"ED"` and `"KPM"` always use it; `"Green"`/`"RG"` pass it on to the Brillouin-zone sum behind the self-energy, where it bites for `gmode="full"` and (in 2D) `gmode="renormalization"` but not for the default `gmode="adaptive"`, whose integration is error-controlled rather than meshed

Whatever the mode, the result is normalized as a density of states: integrating it
over the energies gives the number of states per unit cell, so two modes on the same
system can be compared directly and `h.get_dos()`, the DOS written next to an LDOS
map by `h.get_multildos()`, and the file `dos.dos_ewindow` writes all agree"""))

# --- LDOS section: get_multildos operator and the file it writes
pairs.append((
"""`h.get_multildos()` computes the LDOS at many energies at once, writing one file per energy to a `MULTILDOS/` folder (useful for building an LDOS(x,y,E) movie/stack)

```python
import numpy as np
h.get_multildos(energies=np.linspace(-2.0,2.0,100),projection="atomic")
```
""",
"""`h.get_multildos()` computes the LDOS at many energies at once, writing one file per energy to a `MULTILDOS/` folder (useful for building an LDOS(x,y,E) movie/stack), plus a `MULTILDOS/DOS.OUT` holding the corresponding total DOS

```python
import numpy as np
h.get_multildos(energies=np.linspace(-2.0,2.0,100),projection="atomic")
```

The maps and the `DOS.OUT` beside them carry the same normalization as
`h.get_ldos()` and `h.get_dos()` on the same system, so a map can be read against a
single-energy LDOS and the `DOS.OUT` against `h.get_dos()` without rescaling. An
`operator=` is honoured here as it is by `get_ldos` (the older spelling `op=` still
works, but passing both is a `TypeError`), and the weight it applies is the
expectation value $\\langle\\Psi|A|\\Psi\\rangle$ of the eigenstate -- a gauge-invariant
number -- times the local density, the same convention `get_ldos(mode="arpack")`
uses. `projection="atomic"` does not accept an operator and says so
"""))

apply(pairs)
```

## `e20.py`

```python
from edit import apply
pairs = []

pairs.append((
"""- nk: number of k-points in the mesh. `"ED"` and `"KPM"` always use it; `"Green"`/`"RG"` pass it on to the Brillouin-zone sum behind the self-energy, where it bites for `gmode="full"` and (in 2D) `gmode="renormalization"` but not for the default `gmode="adaptive"`, whose integration is error-controlled rather than meshed

Whatever the mode, the result is normalized as a density of states: integrating it
over the energies gives the number of states per unit cell, so two modes on the same
system can be compared directly and `h.get_dos()`, the DOS written next to an LDOS
map by `h.get_multildos()`, and the file `dos.dos_ewindow` writes all agree""",
"""- nk: number of k-points in the mesh, for `"ED"` and `"KPM"`. `"adaptive"` does not sample a mesh at all -- it integrates over the Brillouin zone with error-controlled quadrature, tuned by `error=1e-1`, and reads `nk` only as a subdivision limit. `"Green"`/`"RG"` pass `nk` to the Brillouin-zone sum behind the self-energy, where it matters for `gmode="full"` and (in 2D) `gmode="renormalization"` but not for the default `gmode="adaptive"`

Whatever the mode, the result is normalized as a density of states: integrating it
over the energies gives the number of states per unit cell. On a spinful chain all
four modes give 2.0 for the same window, so they can be compared with each other
directly, and so can `h.get_dos()`, the DOS written next to an LDOS map by
`h.get_multildos()`, and the file `dos.dos_ewindow` writes"""))

pairs.append((
"""- nk=100: k-points per direction. Used by every mode except `"adaptive"`,
  whose Brillouin-zone integration is error-controlled rather than meshed;
  `"Green"`/`"RG"` forward it to the self-energy's own k-sum, where it
  matters for `gmode="full"` and (in 2D) `gmode="renormalization"`""",
"""- nk=100: k-points per direction, for `"ED"` and `"KPM"`. `"adaptive"`
  integrates with error-controlled quadrature (`error=1e-1`) instead of
  sampling a mesh, and uses `nk` only as a subdivision limit;
  `"Green"`/`"RG"` forward it to the self-energy's own k-sum, where it
  matters for `gmode="full"` and (in 2D) `gmode="renormalization"`"""))

pairs.append((
"""Two restrictions are enforced rather than left to the caller.
`h.get_dos()` accepts only `mode="ED"`, and refuses `use_kpm=True` and
every other mode with a `NotImplementedError`: the Chebyshev and adaptive
expansions both assume a real spectrum. `h.get_ldos()` likewise only
implements `mode="diagonalization"`.""",
"""One restriction is enforced rather than left to the caller: `h.get_dos()`
accepts only `mode="ED"`, and refuses `use_kpm=True` and every other mode
with a `NotImplementedError`, because the Chebyshev and adaptive expansions
both assume a real spectrum. `h.get_ldos()` has the same limitation --
only its default `mode="diagonalization"` is implemented here -- but does
not enforce it, so pass no other mode."""))

pairs.append((
"""`MULTILDOS/` folder, together with a `MULTILDOS/DOS.OUT` and a `DOSMAP.OUT`.""",
"""`MULTILDOS/` folder, together with a `MULTILDOS/DOS.OUT` holding the total
DOS on the same energies and a `DOSMAP.OUT` in the working directory."""))

apply(pairs)
```

## `e21.py`

```python
from edit import apply
apply([(
"""- fermi=0.0: energy below which states are counted as occupied. Not accepted
  for a Nambu Hamiltonian (raises `ValueError`): the Nambu spectrum is
  particle-hole symmetric about zero, so moving the cut does not move the
  electronic energy -- shift the Hamiltonian with `h.shift_fermi()` instead""",
"""- fermi=0.0: energy below which states are counted as occupied. Not accepted
  for a Nambu Hamiltonian (raises `ValueError`): a nonzero `fermi` is not a
  rigid shift of a BdG spectrum, since the electron and hole blocks shift by
  $-\\mu$ and $+\\mu$ -- shift the Hamiltonian instead, with
  `h.shift_fermi(-mu)`, and leave `fermi=0`""")])
```

## `e22.py`

```python
from edit import apply
apply([(
"""The non-Hermitian path re-implements the band structure, the density of
states, the LDOS and the Berry curvature; everything else on a
`non_hermitian=True` Hamiltonian runs the ordinary Hermitian code, which
may or may not be meaningful for a complex spectrum, so check before
relying on it.""",
"""The non-Hermitian path re-implements the band structure, the density of
states, the LDOS and the Berry curvature; everything else on a
`non_hermitian=True` Hamiltonian runs the ordinary Hermitian code, which
may or may not be meaningful for a complex spectrum, so check before
relying on it. The self-consistent mean field is one of those: it is not
re-implemented here, so `h.get_mean_field_hamiltonian()` on a
non-Hermitian Hamiltonian runs the ordinary SCF loop (see
`jupyter-notebooks/functionalities/interacting_mean_field_hamiltonians/08_hermitian_nonhermitian.ipynb`),
and the density matrix it builds is the Hermitian one.""")])
```

## `e23.py`

```python
from edit import apply
apply([(
"""over the energies gives the number of states per unit cell. On a spinful chain all
four modes give 2.0 for the same window, so they can be compared with each other
directly, and so can `h.get_dos()`, the DOS written next to an LDOS map by
`h.get_multildos()`, and the file `dos.dos_ewindow` writes""",
"""over the energies gives the number of states per unit cell. On a spinful chain, whose
answer is 2, all four modes reproduce it to better than 1% on the same window, so
they can be compared with each other directly -- and so can `h.get_dos()`, the DOS
written next to an LDOS map by `h.get_multildos()`, and the file `dos.dos_ewindow`
writes""")])
```

## `e24.py`

```python
from edit import apply
apply([(
"""- `h.get_total_energy(fermi=...)` refuses a Nambu Hamiltonian, where shifting the
  occupation cut does not shift the electronic energy -- use `h.shift_fermi()` on
  the Hamiltonian instead""",
"""- `h.get_total_energy(fermi=...)` refuses a Nambu Hamiltonian, where a nonzero
  `fermi` is not a rigid shift of the spectrum (the electron and hole blocks move
  opposite ways) -- use `h.shift_fermi(-mu)` on the Hamiltonian instead""")])
```

## `e25.py`

```python
from edit import apply
apply([(
"""`-Im Tr G/\\pi` is the density of states, which is what
`h.get_dos(mode="Green")` computes from it""",
"""$-\\mathrm{Im}\\,\\mathrm{Tr}\\,G/\\pi$ is the density of states, which is what
`h.get_dos(mode="Green")` computes from it""")])
```

## `e26.py`

```python
from edit import apply
R="<repo root>/README.md"
apply([(
"""- 0d, 1d, 2d and 3d tight binding models [[notebook]](jupyter-notebooks/functionalities/single_particle_hamiltonians/07_0d_1d_2d_3d_models.ipynb)""",
"""- 0d, 1d, 2d and 3d tight binding models [[notebook]](jupyter-notebooks/functionalities/single_particle_hamiltonians/07_0d_1d_2d_3d_models.ipynb)
- Non-Hermitian Hamiltonians (gain/loss, non-reciprocal hopping), with complex-spectrum band structures, density of states, LDOS and Berry curvature""")],path=R)
```

## `e27.py`

```python
from edit import apply
R="<repo root>/README.md"
apply([(
"""  53 executed notebooks, matching the FUNCTIONALITIES list below -- 53 of its
  74 bullets carry a link to theirs; the rest have no notebook yet""",
"""  53 executed notebooks, matching the FUNCTIONALITIES list below -- 53 of its
  79 bullets carry a link to theirs; the rest have no notebook yet""")],path=R)
```

## `e28.py`

```python
from edit import apply
apply([(
"""n = 20
g = geometry.chain().get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
# a purely imaginary Aubry-Andre modulation: gain on one half of the
# supercell, loss on the other, with no Hermitian part at all
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))""",
"""n = 20
g = geometry.chain().get_supercell(n)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
# a purely imaginary Aubry-Andre modulation, with no Hermitian part at
# all: gain where the cosine is positive, loss where it is negative
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))"""),
("""n = 20
g = geometry.chain().get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))""",
"""n = 20
g = geometry.chain().get_supercell(n)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))""")])
```

## `e29.py`

```python
from edit import apply
apply([(
"""- `h.check()` raises on a Nambu Hamiltonian whose electron-hole symmetry is broken,
  naming the deviation, instead of printing a line and terminating the interpreter""",
"""- `h.check()`, the consistency check on a Hamiltonian, raises naming the deviation
  when a matrix is not Hermitian to within `tol` (1e-5 by default), or when a Nambu
  Hamiltonian's electron-hole symmetry is broken""")])
```

## `e3.py`

```python
from edit import apply
pairs = []

# --- reference: h.get_dos()
pairs.append((
"""### h.get_dos()
Compute the density of states.

Optional arguments:

- energies: array with frequencies of the DOS

- delta=None: broadening of the DOS. Left as `None` it is chosen from the
  k-mesh, `5/nk` (so 0.05 at the default `nk=100`), which keeps the curve
  smooth as the mesh is refined

Return energies and DOS""",
"""### h.get_dos()
Compute the density of states, normalized so that its integral over the
energies is the number of states per unit cell.

Optional arguments:

- energies: array with frequencies of the DOS

- delta=None: broadening of the DOS. Left as `None` it is chosen from the
  k-mesh, `5/nk` (so 0.05 at the default `nk=100`), which keeps the curve
  smooth as the mesh is refined

- mode="ED": `"ED"`, `"KPM"`, `"adaptive"`, `"Green"` or `"RG"` (see the
  "Density of states" section). Anything else raises, listing those five

- nk=100: k-points per direction. Used by every mode except `"adaptive"`,
  whose Brillouin-zone integration is error-controlled rather than meshed;
  `"Green"`/`"RG"` forward it to the self-energy's own k-sum, where it
  matters for `gmode="full"` and (in 2D) `gmode="renormalization"`

- operator=None: operator the DOS is projected onto (a name, a matrix or an
  `Operator`)

For a non-Hermitian Hamiltonian only `mode="ED"` exists, and the extra
`eigmode` argument chooses which part of the complex eigenvalue the
broadening is centred on -- see "Non-Hermitian Hamiltonians".

Return energies and DOS"""))

# --- reference: h.get_multildos()
pairs.append((
"""### h.get_multildos()
Compute the LDOS at many energies, writing one file per energy to a `MULTILDOS/` folder.

Optional arguments:

- energies=linspace(-1,1,100): energies to compute

- projection="TB": `"TB"` or `"atomic"`
""",
"""### h.get_multildos()
Compute the LDOS at many energies, writing one file per energy to a
`MULTILDOS/` folder, together with a `MULTILDOS/DOS.OUT` and a `DOSMAP.OUT`.
The maps and that DOS carry the same normalization as `h.get_ldos()` and
`h.get_dos()`, so they can be read against each other directly.

Optional arguments:

- energies=linspace(-1,1,100): energies to compute

- delta, nk: broadening and k-point density, as in `h.get_ldos()`

- operator=None: operator the LDOS is projected onto, weighting each
  eigenstate by $\\langle\\Psi|A|\\Psi\\rangle$ -- the same convention
  `h.get_ldos(mode="arpack")` uses. The older spelling `op=` is still
  accepted; passing both raises `TypeError`

- projection="TB": `"TB"` or `"atomic"`. Anything else raises `ValueError`
  listing the two, and `"atomic"` together with `operator=` raises
  `NotImplementedError` naming `"TB"` as the projection that supports it

An unrecognized keyword raises `TypeError` rather than being ignored.
"""))

# --- reference: h.get_total_energy()
pairs.append((
"""- fermi=0.0: energy below which states are counted as occupied

- mode="mesh": k-space sampling; `use_kpm=True` switches to a Chebyshev
  estimate for large systems
""",
"""- fermi=0.0: energy below which states are counted as occupied. Not accepted
  for a Nambu Hamiltonian (raises `ValueError`): the Nambu spectrum is
  particle-hole symmetric about zero, so moving the cut does not move the
  electronic energy -- shift the Hamiltonian with `h.shift_fermi()` instead

- mode="mesh": k-space sampling; `use_kpm=True` switches to a Chebyshev
  estimate for large systems

For a Nambu/BdG Hamiltonian this returns the *electronic* energy, i.e. the
energy of the physical electrons rather than of the doubled Nambu spectrum,
so at zero pairing it agrees with the normal-state answer for the same
model. `nbands=` is not implemented there (raises `NotImplementedError`).
Note the caveat under "Spin-spin exchange interactions": the double-counting
correction subtracted by the mean-field total energy is the normal
(Hartree-Fock) one only, never a matching anomalous one, so a total energy
with genuine pairing is still missing that term
"""))

apply(pairs)
```

## `e30.py`

```python
from edit import apply
NEW = r"""## Landauer transmission through a real-space device

`heterostructures` builds a junction out of two periodic Hamiltonians.
`multiterminal.Device` takes the other route: the leads and the scattering
region are given as *geometries*, the hoppings between them are generated
from the actual interatomic distances, and the Landauer transmission
follows from the leads' self-energies. It is the natural form for a
disordered or irregularly-shaped conductor, where there is no unit cell to
repeat.

```python
import numpy as np
from pyqula import geometry
from pyqula import multiterminal

def lead_cell(x): # a one-site chain unit cell sitting at x
    g = geometry.chain() ; g.r = np.array([[x,0.,0.]]) ; g.r2xyz()
    return g

gc = geometry.chain().supercell(4) # the scattering region
xs = gc.r[:,0]
d = multiterminal.Device()
d.biterminal(left_g=lead_cell(min(xs)-1.),right_g=lead_cell(max(xs)+1.),
             central_g=gc,disorder=0.0)
print("ballistic transmission:",d.transmission(energy=0.0)[0])
```

A perfectly matched chain has one open channel and nothing to scatter off,
so the transmission is exactly 1 inside the band -- which is the check
`tests/transport/test_multiterminal_landauer.py` makes. Raising `disorder`
puts random onsite energies in the central region and the transmission
drops below 1 (0.96 at `disorder=1.0` for the four-site region above, on
one particular random realization). `d.transmission(energy=e)` returns the
transmission between every pair of leads, and `multiterminal.landauer` is
the same quantity as a plain function.

## Transport through an arbitrary finite region"""
apply([("## Transport through an arbitrary finite region",NEW)])
```

## `e31.py`

```python
from edit import apply
apply([(
"""one particular random realization). `d.transmission(energy=e)` returns the
transmission between every pair of leads, and `multiterminal.landauer` is
the same quantity as a plain function.""",
"""one particular random realization). `d.transmission(energy=e)` returns a
list with one entry per lead pair, the pair `(0,1)` by default;
`multiterminal.landauer(d,energy,ij=[(i,j)])` is the same quantity as a
plain function, with the pairs chosen explicitly.""")])
```

## `e32.py`

```python
from edit import apply
apply([(
"""A perfectly matched chain has one open channel and nothing to scatter off,
so the transmission is exactly 1 inside the band -- which is the check
`tests/transport/test_multiterminal_landauer.py` makes. Raising `disorder`
puts random onsite energies in the central region and the transmission
drops below 1 (0.96 at `disorder=1.0` for the four-site region above, on
one particular random realization). `d.transmission(energy=e)` returns a""",
"""A perfectly matched chain has one open channel and nothing to scatter off,
so the transmission is 1 everywhere inside the band, to the tolerance of
the lead decimation -- which is the check
`tests/transport/test_multiterminal_landauer.py` makes. Raising `disorder`
puts random onsite energies in the central region and the transmission
drops below 1 by an amount that depends on the realization drawn.
`d.transmission(energy=e)` returns a""")])
```

## `e33.py`

```python
from edit import apply
apply([(
"""```python
h = geometry.chain().get_hamiltonian()
h.add_exchange([0.,0.,0.5])""",
"""```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian()
h.add_exchange([0.,0.,0.5])""")])
```

## `e34.py`

```python
from edit import apply
apply([(
"""```python
hmf = h.get_combined_mean_field_hamiltonian(U=4.0,J1=-0.5,filling=0.5,
        use_jax=True,solver="newton") # JAX-derivative-based SCF solver""",
"""```python
from pyqula import geometry
h = geometry.chain().get_hamiltonian()
hmf = h.get_combined_mean_field_hamiltonian(U=4.0,J1=-0.5,filling=0.5,
        use_jax=True,solver="newton") # JAX-derivative-based SCF solver""")])
```

## `e4.py`

```python
from edit import apply
pairs = []

# --- spin splitting density entry: the collinearity guard
pairs.append((
"""- nk=20: number of k-points per direction
- energies: energies at which the density is evaluated (default 400 points spanning -3 to 3)
- delta=1e-2: broadening
""",
"""- nk=20: number of k-points per direction
- energies: energies at which the density is evaluated (default 400 points spanning -3 to 3)
- delta=1e-2: broadening
- tol=1e-7: largest spin off-diagonal element of the Bloch Hamiltonian
  tolerated, the same guard `get_spin_splitting_vs_energy` applies -- above
  it this raises rather than answering

### h.get_average_spin_splitting()
Return a single number: the spin splitting of a typical band, averaged over
bands as well as over the Brillouin zone. Being an average rather than a
sum it is intensive, so the same crystal described in a larger cell gives
the same answer.

Optional arguments:

- nk=20: number of k-points per direction
- tol=1e-7: collinearity tolerance, as above

Use `h.get_spin_splitting_vs_energy()` instead when what is wanted is the
largest splitting anywhere in the zone rather than the typical one.
"""))

# --- get_berry_curvature operator argument: accepted types
pairs.append((
"""- operator=None: restrict the curvature to a subspace, e.g. `"valley"` for
  a valley-resolved curvature (see "Berry curvature operator")
""",
"""- operator=None: restrict the curvature to a subspace, e.g. `"valley"` for
  a valley-resolved curvature (see "Berry curvature operator"). A name, a
  matrix or an `Operator` are all accepted, the same spellings
  `h.get_bands(operator=...)` takes; the same holds for `h.get_chern()`,
  `topology.chern_density` and `topology.chern_qtci`
"""))

# --- get_chern entry: sparse spin Chern
pairs.append((
"""  curvature is sharply peaked. See "Tensor-cross-interpolation (qtci)
  integration"
""",
"""  curvature is sharply peaked. See "Tensor-cross-interpolation (qtci)
  integration"
- operator=None: a name, a matrix or an `Operator`, as for
  `h.get_berry_curvature()`. The operator-projected invariants
  (`topology.spin_chern`, `topology.operator_berry` and friends) work on
  sparse Hamiltonians too, which is what a moire or supercell model is
"""))

apply(pairs)
```

## `e5.py`

```python
from edit import apply
pairs = []

# --- didv_curve entry: T on a LocalProbe
pairs.append((
"""Arguments:

- energies: array of bias energies
- any keyword argument accepted by `didv` (`method`, `delta`, `use_aaa`, `nmax_max`, `temp`, ...)

Returns an array of dI/dV values
""",
"""Arguments:

- energies: array of bias energies
- any keyword argument accepted by `didv` (`method`, `delta`, `use_aaa`, `nmax_max`, `temp`, ...)

Returns an array of dI/dV values

**`T` means different things on the two classes.** On a `Heterostructure`,
`T` is the temperature. On a `LocalProbe` it is the probe *transparency* --
the same knob as `LocalProbe(...,T=...)`, `set_coupling` and
`get_kappa(T=...)` -- and the temperature there is `temp` (or its alias
`temperature`). Both `lp.didv(T=...)` and `lp.didv_curve(...,T=...)` honour
it for that call alone, leaving the probe itself untouched.

`delta=` passed to any of `didv`, `didv_curve` or `get_smatrix` likewise
applies to that call, self-energies included, rather than being overridden
by the junction's own attribute. For a `LocalProbe` it sets both the probe
broadening and the sample's bulk broadening, exactly as the constructor
argument does. `HT.with_delta(delta)`, `lp.with_delta(delta)` and
`lp.with_coupling(T)` return a shallow copy with that one knob rebound,
for when the same setting is wanted across several calls.
"""))

apply(pairs)
```

## `e6.py`

```python
from edit import apply
pairs = []

pairs.append((
"""`h.get_wannier_hamiltonian()` Wannierizes a fixed, contiguous range of a
periodic Hamiltonian's bands and returns a new, smaller multicell
Hamiltonian whose real-space hoppings exactly reproduce that band subspace
on the wannierization k-mesh (there is no band disentanglement yet -- the
selected range is Wannierized jointly as one group).""",
"""`h.get_wannier_hamiltonian()` Wannierizes a contiguous range of a
periodic Hamiltonian's bands and returns a new, smaller multicell
Hamiltonian whose real-space hoppings reproduce that band subspace on the
wannierization k-mesh. By default the range is taken as a fixed subspace,
Wannierized jointly as one group, and reproduced exactly; passing a
`num_wann` smaller than the range instead disentangles, which is the
subject of the second subsection below."""))

pairs.append((
"""See `examples/wannier/get_wannier_hamiltonian/main.py` and
`examples/wannier/symmetric_wannierization/main.py` for runnable versions
of these two examples.
""",
"""See `examples/wannier/get_wannier_hamiltonian/main.py` and
`examples/wannier/symmetric_wannierization/main.py` for runnable versions
of these two examples.

## Disentanglement

A fixed band range only works when the range is separated from everything
else by a gap across the whole Brillouin zone. Graphene's two $p_z$ bands
are the canonical counterexample: they touch at K, so neither of them is a
smooth subspace on its own. Souza-Marzari-Vanderbilt disentanglement (PRB
65, 035109 (2001)) handles that case by extracting `num_wann` optimally
connected states out of a larger set of *offered* bands, k-point by
k-point. It is switched on by passing a `num_wann` smaller than the
selected range, in which case `bands=[a,b]` only says which bands are
offered.

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)

# one Wannier function out of both pz bands, with the deep part of the
# valence band frozen (kept exactly rather than optimized)
hwan = h.get_wannier_hamiltonian(bands=[0,1],num_wann=1,nk=12,
                                 dis_froz_max=-1.0,cutoff=0.0)

print("Wannier functions:",hwan.wannier_num_wann)
print("window used:",hwan.wannier_disentanglement_window)

f0,fw = h.get_hk_gen(),hwan.get_hk_gen()
k = [2./12.,1./12.,0.] # a wannierization-mesh k-point
e0 = np.sort(np.linalg.eigvalsh(f0(k))) ; ew = np.linalg.eigvalsh(fw(k))
print("frozen state reproduced to:",abs(ew[0]-e0[0]))
```

What is reproduced changes with it, and this is the trade rather than a
loss of accuracy: a disentangled Hamiltonian reproduces the states inside
the **frozen inner window** (`dis_froz_min`/`dis_froz_max`) exactly at
every mesh k-point, and deliberately does not reproduce the selected bands
outside it -- outside the frozen window the extracted subspace is a
different, smoother one. Pass `cutoff=0.0` when testing that exactness, as
above: the default `cutoff=1e-6` drops small real-space hoppings and with
them the last few digits. An **outer window** (`dis_win_min`/
`dis_win_max`) narrows which bands are offered at each k-point, so the
number available varies across the mesh, which is the point of a window
rather than a band range; every eigenvalue of the result then lies inside
it. `dis_num_iter` (default 200) bounds the $\\Omega_I$ minimization --
not reaching convergence within it is a warning rather than an error,
since the frozen window is reproduced either way.

The returned Hamiltonian carries `wannier_num_wann` and
`wannier_disentanglement_window` (the four window values actually used, or
`None` when not disentangling) alongside the centres and spreads.

Combinations that would silently do nothing raise instead: a `dis_*`
window without a smaller `num_wann` (the engine gates disentanglement on
the band count exceeding `num_wann`, so the window would be accepted and
ignored), `dis_froz_min` without `dis_froz_max`, or a `num_wann` outside
`1..len(bands)`. Disentanglement is not implemented together with a
Nambu/BdG Hamiltonian, with `symmetries=`, or with
`auto_split_clusters=True`, each naming the combination. One gap in that
guard is worth knowing: `win_keywords=` is by design an unchecked
passthrough merged after the check, so a window smuggled in through it can
still reach the engine and be ignored -- use the real `dis_froz_max=`
argument, which is guarded.
"""))

# --- the reference entry
pairs.append((
"""### h.get_wannier_hamiltonian()
Wannierize a fixed range of bands and return the resulting real-space
Hamiltonian.

Arguments:

- bands = [a,b]: first and last band to Wannierize (0-indexed, both ends inclusive)

Optional arguments:

- nk=12: k-points per periodic direction for the wannierization mesh
- symmetries=None: `"auto"` to auto-detect and enforce the point group, or an explicit list of `symmetrytk.pointgroup.SymmetryOperation`

Returns a new, smaller Hamiltonian; `.wannier_centres`, `.wannier_spreads` and `.wannier_spread_total` hold the Wannier-function geometry
""",
"""### h.get_wannier_hamiltonian()
Wannierize a range of bands and return the resulting real-space
Hamiltonian.

Arguments:

- bands = [a,b]: first and last band to Wannierize (0-indexed, both ends inclusive)

Optional arguments:

- nk=12: k-points per periodic direction for the wannierization mesh
- cutoff=1e-6: real-space hoppings smaller than this are dropped (the intracell block is always kept). Set it to `0.0` to keep the reproduction exact to machine precision
- symmetries=None: `"auto"` to auto-detect and enforce the point group, or an explicit list of `symmetrytk.pointgroup.SymmetryOperation`
- num_wann=None: how many Wannier functions to extract. `None` means the whole selected range, Wannierized as a fixed subspace and reproduced exactly. A smaller value switches on Souza-Marzari-Vanderbilt disentanglement, after which only the frozen window is reproduced (see "Disentanglement"). Outside `1..len(bands)` raises `ValueError`
- dis_win_min, dis_win_max=None: outer energy window, i.e. which bands are offered to the extraction at each k-point. Every eigenvalue of the result lies inside it
- dis_froz_min, dis_froz_max=None: frozen inner window, whose states are reproduced exactly at every mesh k-point. `dis_froz_max` alone is enough; `dis_froz_min` alone raises `ValueError`
- dis_num_iter=200: maximum iterations of the $\\Omega_I$ minimization; not converging within them is a warning, not an error

Any `dis_*` argument given without a smaller `num_wann` raises `ValueError`
rather than being ignored. Disentanglement together with a Nambu/BdG
Hamiltonian, with `symmetries=`, or with `auto_split_clusters=True` raises
`NotImplementedError`.

Returns a new, smaller Hamiltonian; `.wannier_centres`, `.wannier_spreads`
and `.wannier_spread_total` hold the Wannier-function geometry, and
`.wannier_num_wann`/`.wannier_disentanglement_window` record what was
extracted and through which window (the latter `None` when not
disentangling)
"""))

apply(pairs)
```

## `e7.py`

```python
from edit import apply
R="<repo root>/README.md"
apply([(
"""- Exact reproduction of the selected band subspace on the wannierization mesh [[notebook]](jupyter-notebooks/functionalities/wannierization/02_exact_reproduction.ipynb)""",
"""- Exact reproduction of the selected band subspace on the wannierization mesh [[notebook]](jupyter-notebooks/functionalities/wannierization/02_exact_reproduction.ipynb)
- Souza-Marzari-Vanderbilt band disentanglement, with outer and frozen energy windows""")],path=R)
```

## `e8.py`

```python
from edit import apply
pairs = []
# --- stale claim: J decouples in the normal sector only on a BdG Hamiltonian
pairs.append((
"""On a BdG Hamiltonian, $U$/$V_1$/$V_2$/$V_3$/$V_r$ keep the full
normal+anomalous (pairing) treatment (identical to
`get_mean_field_hamiltonian`), while the exchange ($J$) channels are
decoupled in the normal sector only -- so a state with both magnetic and
superconducting order requires an attractive $V$, not $J$, to seed the
pairing (see the example above).""",
"""On a BdG Hamiltonian every channel keeps the full normal+anomalous
(pairing) treatment: $U$/$V_1$/$V_2$/$V_3$/$V_r$ as in
`get_mean_field_hamiltonian`, and the exchange ($J$) channels identically,
so exchange alone can induce superconducting pairing rather than only
magnetism (see "Spin-spin exchange interactions" above, and the
`total_energy` caveat recorded there)."""))
apply(pairs)
```

## `e9.py`

```python
from edit import apply
pairs = []

NEW = r"""# Non-Hermitian Hamiltonians

An open quantum system -- one exchanging particles or energy with an
environment -- is often described by an effective Hamiltonian that is no
longer Hermitian: gain and loss enter as imaginary onsite energies, and
non-reciprocal hopping ($t_{ij}\ne t_{ji}^*$) as an asymmetric hopping
matrix. The eigenvalues are then complex, their imaginary parts being
amplification and decay rates rather than energies, and the eigenvectors
are no longer orthogonal. This is the setting of photonic and acoustic
lattices with gain and loss, of the non-Hermitian skin effect, and of
$\mathcal{PT}$-symmetric models.

pyqula builds such a Hamiltonian with the `non_hermitian=True` flag, after
which the usual observables route to non-Hermitian implementations
(`pyqula.nonhermitiantk`) instead of the Hermitian ones. Nothing else about
building the model changes -- `add_onsite` with a complex-valued function is
what puts the gain and loss in.

```python
import numpy as np
from pyqula import geometry

n = 20
g = geometry.chain().get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1.])
# a purely imaginary Aubry-Andre modulation: gain on one half of the
# supercell, loss on the other, with no Hermitian part at all
h.add_onsite(lambda r: 0.6j*np.cos(2.*np.pi*r[0]/n))

(ks,es) = h.get_bands(kpath=[[0.,0.,0.]],write=False)
print("eigenvalues are complex:",es.dtype)
print("largest gain rate Im(E):",np.round(np.max(es.imag),4))
```

The eigenvalues come back complex rather than real, so anything that plots
them has to choose a part: `es.real` for the energy axis, `es.imag` for the
gain/decay axis, or both as a scatter in the complex plane.

## Choosing which part is the energy: eigmode

`eigmode` is the argument that exists only on this path. It says which part
of the complex eigenvalue is to play the role of "the energy" -- the
quantity a broadening is centred on, or a band structure is written out
as. It takes `"complex"` (the default, keep the whole eigenvalue),
`"real"`, or `"imag"`; anything else raises `ValueError` listing the three.

In `h.get_bands()` it selects what the returned energy row and the written
`BANDS.OUT` carry. With the default `"complex"` the file gains one extra
column, so its layout is `k`, `Re E`, `Im E`, then one column per operator
-- rather than silently dropping the imaginary part, which is the physics
the calculation was done for.

In `h.get_ldos()` it decides which axis the requested energy `e` lives on.
With `eigmode="imag"` the states are selected by their amplification rate
instead of their energy, which is how one asks where the most amplified
mode of the model above actually sits:

```python
imax = np.argmax(es.imag) # the most amplified state
(x,y,d) = h.get_ldos(e=es[imax].imag,delta=1e-2,eigmode="imag",nrep=1)
print("it lives at x =",x[np.argmax(d)])
```

which returns the site where the gain is largest, $x=-0.5$ here -- the
maximum of the modulation above.

`h.get_dos()` broadens the real part whatever `eigmode` says, so
`"complex"` and `"real"` give the same density of states there and
`"imag"` gives the distribution of decay rates instead.

## What is and is not available

The non-Hermitian path re-implements the band structure, the density of
states, the LDOS and the Berry curvature; everything else on a
`non_hermitian=True` Hamiltonian runs the ordinary Hermitian code, which
may or may not be meaningful for a complex spectrum, so check before
relying on it.

Two restrictions are enforced rather than left to the caller.
`h.get_dos()` accepts only `mode="ED"`, and refuses `use_kpm=True` and
every other mode with a `NotImplementedError`: the Chebyshev and adaptive
expansions both assume a real spectrum. `h.get_ldos()` likewise only
implements `mode="diagonalization"`.

Operators work as usual, including `operator="unfold"`, so a supercell
calculation can be unfolded back onto the primitive Brillouin zone with a
complex spectrum -- see `examples/1d/unfolding_non_hermitian/main.py`.
`num_bands` also works, selecting the few eigenvalues nearest
`central_energy` with ARPACK rather than diagonalizing fully.

See `examples/1d/NH_ldos/main.py` (the model above, resolved mode by
mode), `examples/0d/non_hermitian_aah/main.py` and
`examples/0d/non_hermitian_aah_dos/main.py` (a non-Hermitian Aubry-Andre
chain swept over the modulation phase), and `tests/nonhermitian/` for the
invariants these paths are held to.

# Superconductivity"""

pairs.append(("# Superconductivity",NEW))

pairs.append((
"""- [Operators](#operators)
- [Superconductivity](#superconductivity)""",
"""- [Operators](#operators)
- [Non-Hermitian Hamiltonians](#non-hermitian-hamiltonians)
- [Superconductivity](#superconductivity)"""))

apply(pairs)
```

## `edit.py`

```python
import sys,io
G="<repo root>/documentation/user_guide.md"
def apply(pairs,path=G):
    s=open(path).read()
    for old,new in pairs:
        n=s.count(old)
        if n!=1:
            raise SystemExit("ANCHOR count %d for:\n%r"%(n,old[:200]))
        s=s.replace(old,new)
    open(path,"w").write(s)
    print("ok, %d edits"%len(pairs))
```

## `entrypoints.py`

```python
import ast,os,subprocess,re,sys
ROOT="<repo root>"
def methods(path,cls):
    t=ast.parse(open(path).read())
    out=[]
    for n in ast.walk(t):
        if isinstance(n,ast.ClassDef) and n.name==cls:
            for m in n.body:
                if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and not m.name.startswith("_"):
                    out.append((m.name,m.lineno))
    return out
ham=methods(ROOT+"/src/pyqula/hamiltonians.py","Hamiltonian")
geo=methods(ROOT+"/src/pyqula/geometry.py","Geometry")
# gather all text of tests and examples
def blob(d):
    s=[]
    for dp,_,fns in os.walk(ROOT+"/"+d):
        for f in fns:
            if f.endswith(".py"):
                s.append(open(os.path.join(dp,f),errors="ignore").read())
    return "\n".join(s)
T=blob("tests"); E=blob("examples"); D=open(ROOT+"/documentation/user_guide.md",errors="ignore").read()
print("=== Hamiltonian methods with NO occurrence in tests/ ===")
for (m,l) in ham:
    nt=T.count(m); ne=E.count(m); nd=D.count(m)
    if nt==0:
        print(f"  {m:45s} line {l:5d} tests={nt} examples={ne} guide={nd}")
print("=== Hamiltonian methods in tests but 0 examples and 0 guide ===")
for (m,l) in ham:
    nt=T.count(m); ne=E.count(m); nd=D.count(m)
    if nt>0 and ne==0 and nd==0 and nt<3:
        print(f"  {m:45s} line {l:5d} tests={nt}")
print("=== Geometry methods with NO occurrence in tests/ ===")
for (m,l) in geo:
    nt=T.count(m); ne=E.count(m); nd=D.count(m)
    if nt==0:
        print(f"  {m:45s} line {l:5d} tests={nt} examples={ne} guide={nd}")
```

## `extract.py`

```python
import os,re,sys
GUIDE="<repo root>/documentation/user_guide.md"
OUT="SCRATCH"
lines=open(GUIDE).read().split("\n")
sec="preamble"; secline=0
blocks=[]  # (section, startline, code)
i=0
while i<len(lines):
    l=lines[i]
    if l.startswith("#") and not l.startswith("#!"):
        sec=l.strip("# ").strip(); secline=i+1
    if l.strip().startswith("```python"):
        j=i+1; code=[]
        while j<len(lines) and not lines[j].strip().startswith("```"):
            code.append(lines[j]); j+=1
        blocks.append((sec,i+1,"\n".join(code)))
        i=j
    i+=1
# group consecutive blocks in the same section: cumulative script per block
cur=None; acc=[]
files=[]
for n,(sec,ln,code) in enumerate(blocks):
    if sec!=cur:
        cur=sec; acc=[]
    acc.append(code)
    name="b%03d_L%04d_%s.py"%(n,ln,re.sub(r'[^A-Za-z0-9]+','_',sec)[:40])
    with open(os.path.join(OUT,name),"w") as f:
        f.write("# section: %s   guide line %d\n"%(sec,ln))
        f.write("import matplotlib\nmatplotlib.use('Agg')\n")
        f.write("\n\n".join(acc)+"\n")
    files.append((name,sec,ln))
for f in files: print(f[0],"|",f[1],"| L",f[2])
```

## `extract2.py`

```python
import re,os
R="<repo root>/"
src=open(R+"documentation/user_guide.md").read().split("\n")
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"snips2")
os.makedirs(out,exist_ok=True)
blocks=[];cur=None;start=0;sec=("?",0)
for i,l in enumerate(src):
    if (l.startswith("# ") or l.startswith("## ")) and cur is None: sec=(l,i)
    if l.strip()=="```python" and cur is None: cur=[];start=i+1;cursec=sec
    elif l.strip()=="```" and cur is not None:
        blocks.append((start+1,cursec,"\n".join(cur)));cur=None
    elif cur is not None: cur.append(l)
# group by section start line
for n,(ln,s,b) in enumerate(blocks):
    prev=[bb for (lnn,ss,bb) in blocks if ss==s and lnn<ln]
    txt="# cumulative for user_guide.md line %d, section %s\n"%(ln,s[0])+"\n".join(prev+[b])+"\n"
    open(os.path.join(out,"c%03d_L%04d.py"%(n,ln)),"w").write(txt)
print(len(blocks))
```

## `finiteT2.py`

```python
import numpy as np, os
from pyqula import geometry, meanfield
def ne(scf): return np.trace(np.array(scf.dm[(0,0,0)])).real
g = geometry.chain()
print("filling   T      requested N  converged N   rel.error")
for filling,T in [(0.1,0.05),(0.1,0.1),(0.1,0.2),(0.2,0.1),(0.4,0.2),(0.45,0.3)]:
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = g.get_hamiltonian()
    s = meanfield.Vinteraction(h,U=1.0,filling=filling,T=T,nk=60,mf="ferroZ",
            mix=0.3,maxerror=1e-7,maxite=400,load_mf=False,verbose=0)
    N0 = 2*filling
    print(" %.2f    %-5.3g  %.6f     %.6f   %+.2f%%"%(filling,T,N0,ne(s),100*(ne(s)-N0)/N0))
```

## `finiteT_control.py`

```python
import numpy as np, os
from pyqula import geometry, meanfield
def ne(s): return np.trace(np.array(s.dm[(0,0,0)])).real
g=geometry.chain()
for T in (1e-7,1e-3,0.01,0.05):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h=g.get_hamiltonian()
    s=meanfield.Vinteraction(h,U=1.0,filling=0.1,T=T,nk=60,mf="ferroZ",mix=0.3,
        maxerror=1e-7,maxite=400,load_mf=False,verbose=0)
    print("filling=0.10 T=%-8.4g requested N=0.200000  converged N=%.6f  err=%+.2f%%"%(T,ne(s),100*(ne(s)-0.2)/0.2))
```

## `finiteT_filling.py`

```python
"""The SCF's Fermi level is found by a T=0 eigenvalue count
(spectrum.get_fermi4filling -> filling.get_fermi_energy, no T argument at all),
but the density matrix at that Fermi level is built with Fermi-Dirac at T.
Away from a particle-hole-symmetric point the converged electron count then
drifts away from the requested filling as T grows."""
import numpy as np, os
from pyqula import geometry, meanfield

def ne(scf):  # electrons per cell from the converged density matrix
    return np.trace(np.array(scf.dm[(0,0,0)])).real

g = geometry.chain()
print(" engine          filling   T       requested N   converged N   drift")
for T in (1e-7,0.05,0.2,0.5):
  for filling in (0.3,):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = g.get_hamiltonian()
    s = meanfield.Vinteraction(h,U=1.0,filling=filling,T=T,nk=40,mf="ferroZ",
            mix=0.3,maxerror=1e-7,maxite=400,load_mf=False,verbose=0)
    N0 = 2*filling
    print(" Vinteraction     %.2f   %-7.4g  %.6f      %.6f   %+.4f"%(filling,T,N0,ne(s),ne(s)-N0))
    h = g.get_hamiltonian()
    s2 = meanfield.VJinteraction(h,U=1.0,filling=filling,T=T,nk=40,mf="ferroZ",
            mix=0.3,maxerror=1e-7,maxite=400,verbose=0)
    print(" VJinteraction    %.2f   %-7.4g  %.6f      %.6f   %+.4f"%(filling,T,N0,ne(s2),ne(s2)-N0))
```

## `fixture.py`

```python
"""Non-Hermitian, non-symmetric inter-cell coupling fixture, 2 orbitals, 1D."""
import numpy as np
from pyqula import geometry

def make_T(seed=1,norb=2,scale=0.6):
    rs = np.random.RandomState(seed)
    T = scale*(rs.randn(norb,norb)+1j*rs.randn(norb,norb))
    return T

def make_H0(seed=2,norb=2,scale=0.4):
    rs = np.random.RandomState(seed)
    A = scale*(rs.randn(norb,norb)+1j*rs.randn(norb,norb))
    return A+A.conj().T

def build_h(T=None,H0=None,norb=2):
    """1D Hamiltonian with `norb` orbitals per cell and generic complex T."""
    if T is None: T = make_T(norb=norb)
    if H0 is None: H0 = make_H0(norb=norb)
    g = geometry.chain()
    g = g.get_supercell(norb) if norb>1 else g
    h = g.get_hamiltonian(has_spin=False)
    from pyqula.multihopping import MultiHopping
    d = {(0,0,0):np.array(H0,dtype=np.complex128),
         (1,0,0):np.array(T,dtype=np.complex128),
         (-1,0,0):np.array(T,dtype=np.complex128).conj().T}
    h.set_multihopping(MultiHopping(d))
    return h

def checks(T):
    return dict(dagger=np.max(np.abs(T-T.conj().T)),
                transpose=np.max(np.abs(T-T.T)),
                conj=np.max(np.abs(T-T.conj())))

if __name__=="__main__":
    T = make_T(); H0 = make_H0()
    print("T=",T); print("checks:",checks(T))
    h = build_h(T,H0)
    print("is_multicell",h.is_multicell,"dim",h.dimensionality)
    h2 = h.get_no_multicell()
    print("intra ok:",np.max(np.abs(h2.intra-H0)))
    print("inter vs T:",np.max(np.abs(h2.inter-T)),"  inter vs T^dag:",np.max(np.abs(h2.inter-T.conj().T)))
    hk = h.get_hk_gen()
    k = 0.31
    ref = H0 + T*np.exp(1j*2*np.pi*k) + T.conj().T*np.exp(-1j*2*np.pi*k)
    m = np.array(hk([k,0,0]))
    print("hk vs H0+T e^{+ik}+Td e^{-ik}:",np.max(np.abs(m-ref)))
    ref2 = H0 + T*np.exp(-1j*2*np.pi*k) + T.conj().T*np.exp(1j*2*np.pi*k)
    print("hk vs H0+T e^{-ik}+Td e^{+ik}:",np.max(np.abs(m-ref2)))
    print("hk hermitian:",np.max(np.abs(m-m.conj().T)))
```

## `importsweep.py`

```python
import pkgutil,importlib,sys,traceback
sys.path.insert(0,"<repo root>/src")
import pyqula
bad=[]
for m in pkgutil.walk_packages(pyqula.__path__,"pyqula."):
    n=m.name
    if "qutecipytk" in n: continue
    try: importlib.import_module(n)
    except BaseException as e:
        bad.append((n,type(e).__name__,str(e)[:200]))
for b in bad: print(b)
print("TOTAL BAD",len(bad))
```

## `kondo_branches.py`

```python
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
J=1.5 ; filling=0.15 ; nk=200
def mk():
    gc=geometry.chain(); return KondoLatticeHamiltonian(gc.get_hamiltonian(has_spin=True))
hK,eK = mk().get_mean_field_hamiltonian(J=J,filling=filling,nk=nk,
        mf=(np.array([0.3+0j]),np.array([0.0])),mix=0.3,maxerror=1e-8,maxite=5000,
        return_total_energy=True)
hT,eT = mk().get_mean_field_hamiltonian(J=J,filling=filling,nk=nk,mix=0.3,
        maxerror=1e-8,maxite=500,return_total_energy=True)
V = hK.hybridization[0]
corr = np.sum(np.abs(hK.hybridization)**2)/J   # the MISSING second copy
print("Kondo branch : reported E = %.8f   corrected (E + N|V|^2/J extra) = %.8f"%(eK,eK+corr))
print("trivial V=0  : reported E = %.8f   corrected = %.8f"%(eT,eT))
print("reported  dE = %.8f   corrected dE = %.8f"%(eK-eT, eK+corr-eT))
print("missing HS term |V|^2/J = %.8f   (%.1f%% of |E_kondo|)"%(corr,100*corr/abs(eK)))
```

## `kondo_stat2.py`

```python
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.multihopping import MultiHopping

J=1.5 ; filling=0.15 ; nk=200 ; T=2e-2 ; Q=1.0 ; Jg=J/2.
gc = geometry.chain(); hc = gc.get_hamiltonian(has_spin=True)
K = KondoLatticeHamiltonian(hc)
pairs = K._kondo_pairs
h1 = K.get_dense()
hop0 = h1.get_dict()
mu = h1.get_fermi4filling(filling, nk=nk)

def build_extra(V,lam):
    m = np.zeros(h1.intra.shape,dtype=np.complex128)
    for idx,(ci,fi) in enumerate(pairs):
        for s in (0,1):
            cc,ff = 2*ci+s, 2*fi+s
            m[ff,ff] += lam[idx]; m[cc,ff] += np.conjugate(V[idx]); m[ff,cc] += V[idx]
    return {(0,0,0):m}

def omega(V,lam,c):
    hop = MultiHopping(hop0)+MultiHopping(build_extra(V,lam))
    h = h1.copy(); h.set_multihopping(hop); h.fermi = mu; h.shift_fermi(-mu)
    dm = h.get_density_matrix(nk=nk,T=T,ds=[(0,0,0)])[(0,0,0)]
    A = np.array([dm[2*fi,2*ci]+dm[2*fi+1,2*ci+1] for (ci,fi) in pairs])
    eband = h.get_total_energy(nk=nk)          # sum_occ (e - mu)
    return eband + c*np.sum(np.abs(V)**2)/J - np.sum(lam)*Q, A

seed=(np.array([0.3+0j]),np.array([0.0]))
h2 = K.get_mean_field_hamiltonian(J=J,filling=filling,nk=nk,mf=seed,mix=0.3,
        maxerror=1e-8,maxite=5000)
Vstar = h2.hybridization.copy(); lam = h2.constraint_lambda.copy()
print("SCF: V*=%.8f  lam=%.8f  n_f=%.6f"%(Vstar[0].real,lam[0],h2.local_occupation[0]))
O,A = omega(Vstar,lam,1.0)
print("At the fixed point: A=<f^dag c>=%s ;  -J/2*A = %.8f  (== V*, the SCF condition)"
        %(np.round(A[0],8), (-Jg*A[0]).real))
print("   dOmega/dV with c=1 :  A + 1*V/J = %.6f"%( (A[0]+1*Vstar[0]/J).real))
print("   dOmega/dV with c=2 :  A + 2*V/J = %.6e"%( (A[0]+2*Vstar[0]/J).real))
print()
print("  V        Omega(c=1)        Omega(c=2)")
for x in np.linspace(0.30,0.46,17):
    V = np.array([x+0j])
    o1,_ = omega(V,lam,1.0); o2,_ = omega(V,lam,2.0)
    star = " <-- SCF V*" if abs(x-Vstar[0].real)<0.006 else ""
    print("%6.3f  %14.8f  %14.8f%s"%(x,o1.real,o2.real,star))
```

## `kondo_stationarity.py`

```python
"""The Kondo-lattice mean-field grand potential must be STATIONARY in V at
the SCF fixed point (Hellmann-Feynman):

   Omega(V) = sum_occ (e_n - mu)  +  c*|V|^2/J  -  lam*Q
   dOmega/dV = <f^dag c>_summed_over_spin + c*V^*/J  =  A + c V^*/J

The SCF's own fixed point is V = -(J/2)*A, i.e. A = -2V/J, so
   dOmega/dV = (c-2) V / J   (for real V)
which vanishes only for c=2 (Coleman Eq. 78's N*Vbar*V/J with N=2).
scftk/kondolattice.py:_pack uses c=1 (`np.sum(np.abs(V)**2)/J`).

This script scans Omega(V) with c=1 and c=2 around the SCF's V* and reports
where each is minimised.
"""
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.multihopping import MultiHopping

J=1.5 ; filling=0.15 ; nk=200 ; T=2e-2 ; Q=1.0
gc = geometry.chain()
hc = gc.get_hamiltonian(has_spin=True)
K = KondoLatticeHamiltonian(hc)
seed = (np.array([0.3+0j]), np.array([0.0]))
h2 = K.get_mean_field_hamiltonian(J=J, filling=filling, nk=nk, mf=seed,
        mix=0.3, maxerror=1e-8, maxite=5000)
assert h2 is not None
Vstar = h2.hybridization[0].real ; lam = h2.constraint_lambda[0]
print("SCF fixed point: V* = %.8f   lam = %.8f   n_f = %.6f"%(Vstar,lam,h2.local_occupation[0]))

# rebuild the same one-body problem by hand
h1 = K.get_bare_hamiltonian().get_dense() if hasattr(K,"get_bare_hamiltonian") else None
# fall back: reconstruct from the module internals
from pyqula.scftk import kondolattice as KL
import inspect
h0 = K.h0 if hasattr(K,"h0") else None
print("attrs:",[a for a in dir(K) if not a.startswith("_")][:40])
```

## `lens5_swallow.py`

```python
import ast, os
ROOT="<repo root>/src/pyqula"
FENCE=("scftk","meanfield.py","sctk","superconductivity.py","greentk","transporttk",
       "keldyshtk","aaatk","qtcitk","heterostructures.py","topology.py","topologytk",
       "multicell.py","gauge.py","qutecipytk")
def fenced(rel):
    if rel.startswith("wanniertk/wannierpy"): return True
    p=rel.split(os.sep)
    for f in FENCE:
        if f.endswith(".py"):
            if rel==f: return True
        elif p[0]==f: return True
    return False
class NC(ast.NodeVisitor):
    def __init__(self): self.n=set()
    def visit_Name(self,x): self.n.add(x.id); self.generic_visit(x)
    def visit_Attribute(self,x): self.n.add(x.attr); self.generic_visit(x)
    def visit_keyword(self,x):
        if x.arg: self.n.add(x.arg)
        self.generic_visit(x)
    def visit_arg(self,x): self.n.add(x.arg); self.generic_visit(x)
out=[]
for dp,ds,fs in os.walk(ROOT):
    for f in fs:
        if not f.endswith(".py"): continue
        p=os.path.join(dp,f); rel=os.path.relpath(p,ROOT)
        if fenced(rel): continue
        try: t=ast.parse(open(p).read())
        except Exception: continue
        for n in ast.walk(t):
            if not isinstance(n,ast.FunctionDef): continue
            if n.args.kwarg is None: continue
            nc=NC()
            for s in n.body: nc.visit(s)
            if n.args.kwarg.arg not in nc.n:
                out.append(f"{rel}:{n.lineno} {n.name}  swallows **{n.args.kwarg.arg}")
print("\n".join(sorted(out)))
print("TOTAL",len(out))
```

## `methodcoverage.py`

```python
import subprocess, ast, os
R="<repo root>"
def methods(path, cls):
    tree=ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n,ast.ClassDef) and n.name==cls:
            return [(m.name,m.lineno) for m in n.body if isinstance(m,ast.FunctionDef) and not m.name.startswith("_")]
    return []
out=[]
for path,cls in [(R+"/src/pyqula/hamiltonians.py","Hamiltonian"),(R+"/src/pyqula/geometry.py","Geometry")]:
    for name,ln in methods(path,cls):
        hits=subprocess.run(["grep","-rl","."+name+"(",R+"/tests",R+"/examples"],capture_output=True,text=True).stdout.split()
        if not hits:
            out.append((cls,name,ln))
for c,n,l in out: print(f"{c}.{n}  (line {l})")
print("UNCOVERED",len(out))
```

## `nambu_etot.py`

```python
"""A Nambu Hamiltonian with a REPULSIVE V1 converges to zero pairing, so its
total_energy must equal the non-Nambu Vinteraction total_energy for the same
system (this is NOT the known anomalous-double-counting gap: there is no
anomalous mean field here at all)."""
import numpy as np, os
from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector
g = geometry.chain()
for filling in (0.3,0.5):
  for V1 in (1.0,2.0):
    res={}
    for nambu in (False,True):
        if os.path.exists("MF.pkl"): os.remove("MF.pkl")
        h = g.get_hamiltonian()
        if nambu: h.setup_nambu_spinor()
        h0=h.copy()
        s = meanfield.Vinteraction(h,V1=V1,filling=filling,nk=40,mf="ferroZ",
                mix=0.3,maxerror=1e-8,maxite=1000,load_mf=False,verbose=0)
        mfm = np.array(s.hamiltonian.intra)-np.array(h0.intra)
        pair = np.max(np.abs(get_eh_sector(mfm,i=0,j=1))) if nambu else 0.
        res[nambu]=(s.total_energy,pair,s.converged)
    print("filling=%.2f V1=%.1f : normal E=%.8f   nambu E=%.8f   diff=%+.8f  (nambu pairing=%.1e)"%(
        filling,V1,res[False][0],res[True][0],res[True][0]-res[False][0],res[True][1]))
```

## `nambu_etot2.py`

```python
"""BdG total_energy with EXACTLY ZERO pairing (repulsive V1) does not match the
normal-state total_energy for the same system.  Decomposed against the exact
BdG identity  sum_{E<0} E_BdG = 2*E_normal - Tr h  verified in bdg_band_energy.py."""
import numpy as np, os
from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector
from pyqula.scftk.densitydensity import get_dc_energy

g = geometry.chain(); V1=1.0; filling=0.5; nk=40
out={}
for nambu in (False,True):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = g.get_hamiltonian()
    if nambu: h.setup_nambu_spinor()
    s = meanfield.Vinteraction(h,V1=V1,filling=filling,nk=nk,mf="ferroZ",
            mix=0.3,maxerror=1e-9,maxite=2000,load_mf=False,verbose=0)
    hc = s.hamiltonian
    band = hc.get_total_energy(nk=nk)
    mun  = hc.fermi*hc.intra.shape[0]*filling
    dmdc = s.dm
    if nambu: dmdc = {k:get_eh_sector(m,i=0,j=0) for k,m in s.dm.items()}
    dc = get_dc_energy(s.v,dmdc).real
    out[nambu]=(band,mun,dc,s.total_energy,hc.fermi,hc.intra.shape[0],np.trace(np.array(hc.intra)).real)
    lbl = "NAMBU " if nambu else "normal"
    print("%s band=%12.8f  muN=%12.8f  dc=%12.8f  -> total=%12.8f  (fermi=%.6f, dim=%d, Tr h=%.6f)"%(
        lbl,band,mun,dc,s.total_energy,hc.fermi,hc.intra.shape[0],np.trace(np.array(hc.intra)).real))
bn,mn,dn,tn,fn,dimn,trn = out[False]
bb,mb,db,tb,fb,dimb,trb = out[True]
print()
print("band:  reported BdG %.8f   vs 2*normal - Tr h = %.8f   (exact BdG identity)"%(bb,2*bn-trb))
print("muN :  reported BdG %.8f   vs 2*normal muN    = %.8f   (consistently doubled)"%(mb,2*mn))
print("dc  :  reported BdG %.8f   vs normal dc       = %.8f   (NOT doubled)"%(db,dn))
print()
print("=> BdG total = 2*(normal band+muN) - Tr h + dc = %.8f ; reported %.8f"%(2*(bn+mn)-trb+db,tb))
print("   normal total = %.8f    discrepancy = %.8f"%(tn,tb-tn))
```

## `p1b_convention.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
T=make_T();H0=make_H0()
h=build_h(T,H0)
hs=h.get_supercell(3).get_no_multicell()
mm=hs.intra
m=np.array(mm.todense()) if hasattr(mm,'todense') else np.array(mm)
print("supercell intra shape",m.shape)
# block (0,1) of the supercell intra -> is it T or T^dag?
b01=m[0:2,2:4]; b10=m[2:4,0:2]
print("block(0,1)-T:",np.max(np.abs(b01-T)),"  block(0,1)-Td:",np.max(np.abs(b01-T.conj().T)))
print("block(1,0)-T:",np.max(np.abs(b10-T)),"  block(1,0)-Td:",np.max(np.abs(b10-T.conj().T)))
```

## `p2_gaussinv.py`

Backs: **L3 transport dagger** — CLEARED: gauss_inverse (the Gauss block-tridiagonal inverse) on non-Hermitian, non-dagger-related couplings

```python
import numpy as np
np.random.seed(3)
from pyqula.algebratk.gaussinv import gauss_inverse as ginv
from pyqula.green import block_inverse

def rnd(n,m=None):
    if m is None: m=n
    return np.random.randn(n,m)+1j*np.random.randn(n,m)

for norb in [1,2,3]:
  for nb in [3,4,5]:
    m = [[None]*nb for _ in range(nb)]
    for i in range(nb): m[i][i] = rnd(norb)
    for i in range(nb-1):
        m[i][i+1] = rnd(norb)          # generic, NOT dagger-related
        m[i+1][i] = rnd(norb)
    worst=0.; worstij=None
    for i in range(nb):
      for j in range(nb):
        a = np.array(ginv(m,i=i,j=j))
        b = np.array(block_inverse(m,i=i,j=j))
        e = np.max(np.abs(a-b))/max(1e-30,np.max(np.abs(b)))
        if e>worst: worst,worstij=e,(i,j)
    print("norb=%d nb=%d  worst rel err=%.3e at (i,j)=%s"%(norb,nb,worst,worstij))
```

## `p2b_nonuniform.py`

Backs: **L3 transport dagger** — gauss_inverse cannot return off-diagonal blocks when the block sizes differ, although landauer's own comment advertises that support

```python
import numpy as np
np.random.seed(5)
from pyqula.algebratk.gaussinv import gauss_inverse as ginv
from pyqula.green import block_inverse
def rnd(n,m=None):
    if m is None: m=n
    return np.random.randn(n,m)+1j*np.random.randn(n,m)
sizes=[2,3,2]
nb=len(sizes)
m=[[None]*nb for _ in range(nb)]
for i in range(nb): m[i][i]=rnd(sizes[i])
for i in range(nb-1):
    m[i][i+1]=rnd(sizes[i],sizes[i+1])
    m[i+1][i]=rnd(sizes[i+1],sizes[i])
print("block sizes",sizes)
try:
    a=ginv(m,i=nb-1,j=0); print("gauss ok, shape",a.shape)
except Exception as e:
    print("gauss RAISED:",type(e).__name__,e)
b=block_inverse(m,i=nb-1,j=0); print("block_inverse shape",b.shape)
print("rel err (n-1,0):",np.max(np.abs(a-b))/np.max(np.abs(b)))
for (i,j) in [(0,0),(0,1),(1,0),(0,2),(2,0),(1,2),(2,1),(2,2)]:
    try:
        x=np.array(ginv(m,i=i,j=j)); y=np.array(block_inverse(m,i=i,j=j))
        print((i,j),"relerr %.2e"%(np.max(np.abs(x-y))/np.max(np.abs(y))))
    except Exception as e:
        print((i,j),"RAISED",type(e).__name__,e)
```

## `perf1_extract_triplet_pairing_loop.py`

Backs: **L2 SC observables** — The four pairing-extraction kernels are O(nsites^2) interpreted Python double loops, re-run at every k-point

```python
import numpy as np, time
import pyqula.superconductivity
from pyqula import geometry
import pyqula.sctk.extract as ex
import pyqula.sctk.dvector as dv

orig = ex.extract_triplet_pairing
calls = [0]
def counted(m):
    calls[0] += 1
    return orig(m)
def vectorized(m):
    from pyqula import algebra
    m = np.asarray(algebra.todense(m))
    return (m[0::4,3::4], m[1::4,2::4], (m[0::4,2::4]-np.conjugate(m[3::4,1::4].T))/2.)

g = geometry.honeycomb_lattice().supercell(5)   # 50 sites
h = g.get_hamiltonian()
h.add_pairing(delta=0.3, mode="triplet", d=[1.,1j,0.])
print("sites",len(g.r),"matrix",h.intra.shape)

dv.extract = ex
ex.extract_triplet_pairing = counted
t0=time.time(); q1 = h.get_dvector_non_unitarity(nk=6); t1=time.time()
print("loop version    %.3f s, extract_triplet_pairing calls = %d"%(t1-t0,calls[0]))
ex.extract_triplet_pairing = vectorized
t2=time.time(); q2 = h.get_dvector_non_unitarity(nk=6); t3=time.time()
print("strided version %.3f s  ratio %.0fx"%(t3-t2,(t1-t0)/(t3-t2)))
print("max |difference| =", np.max(np.abs(q1-q2)))
```

## `perf2_superfluidweight_einsum.py`

Backs: **L2 SC observables** — Three-operand np.einsum with default optimize=False in the superfluid-weight diamagnetic term bypasses BLAS

```python
import numpy as np, time
from pyqula import geometry
import pyqula.sctk.superfluidweight as sfw
from pyqula.sctk.superfluidweight import _divided_difference, _fermi

def patched(es,ws,A,B,T,nd):
    W = _divided_difference(es,es,T); wsc = np.conjugate(ws)
    Arot = [wsc.T@a@ws for a in A]
    para = np.zeros((nd,nd))
    for a in range(nd):
        for b in range(a,nd):
            v = np.sum(W*Arot[a]*Arot[b].T).real; para[a,b]=v; para[b,a]=v
    nf = _fermi(es,T); dia = np.zeros((nd,nd))
    for (a,b) in B:
        if a>b: continue
        v = np.sum(nf*np.sum(wsc*(B[(a,b)]@ws),axis=0)).real   # gemm form
        dia[a,b]=v; dia[b,a]=v
    return para,dia

g = geometry.honeycomb_lattice().supercell(4)   # 32 sites -> 128x128 BdG
h = g.get_hamiltonian(); h.add_onsite(-0.4); h.add_swave(0.3)
print("BdG matrix",h.intra.shape)
orig = sfw._superfluid_weight_at
t0=time.time(); D1 = h.get_superfluid_weight(nk=6); t1=time.time()
sfw._superfluid_weight_at = patched
t2=time.time(); D2 = h.get_superfluid_weight(nk=6); t3=time.time()
sfw._superfluid_weight_at = orig
print("as shipped  %.3f s"%(t1-t0))
print("gemm form   %.3f s   ratio %.1fx"%(t3-t2,(t1-t0)/(t3-t2)))
print("max |D1-D2| =",np.max(np.abs(D1-D2)), " D =",np.round(D1[0,0],8))
```

## `perf_dm_sparse_pairs.py`

Backs: **L6 optimization** — The main density-density SCF loop computes a fully dense per-direction density matrix; the sparse-pairs kernel proven for VJinteraction is never used there

```python
"""LENS6 perf: the sparse-pairs density matrix proven for VJinteraction
(commit 92ff79d: densitymatrix.full_dm_accumulate_sparse +
dmtk.fulldm.full_dm_batch_d_sparse) is NOT used by the main
density-density SCF path (scftk.densitydensity.generic_densitydensity ->
get_dm(integration='ed') -> h.get_density_matrix(ds=...) ->
densitymatrix.full_dm_accumulate, the DENSE per-direction kernel).
Oracle: zero every dm entry outside v[d]'s nonzero pattern (keeping the
(0,0,0) diagonal) -- the mean field must be UNCHANGED, i.e. the dense
kernel computes n^2 entries per (k,direction) that nothing reads."""
import os, tempfile, numpy as np
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, specialhopping
from pyqula.scftk.densitydensity import get_mf_normal, get_dm

g = geometry.honeycomb_lattice().supercell(3)
h = g.get_hamiltonian(has_spin=False).get_multicell().get_dense()
n = h.intra.shape[0]
nd = g.neighbor_distances()
mgen = specialhopping.distance_hopping_matrix([1.0/2.,0.,0.],nd[0:3])
hv = g.get_hamiltonian(has_spin=False,is_multicell=True,mgenerator=mgen)
v = hv.get_hopping_dict()
print("n =",n," number of interaction directions:",len(v))
tot_nnz = 0
for d in sorted(v):
    nz = int(np.count_nonzero(v[d])); tot_nnz += nz
    print("  dir",d," nnz(v[d]) =",nz," of n^2 =",n*n,
          " (%.1f%%)"%(100.*nz/(n*n)))
print("entries the dense kernel computes per k: len(ds)*n^2 =",
      (len(v)+1)*n*n, " ; entries actually read: ~",tot_nnz+n)

dm = get_dm(h, v, nk=4, integration="ed")
mf_dense = get_mf_normal(v, dm)

dm_masked = {}
for d in dm:
    dneg = (-d[0],-d[1],-d[2])
    mask = np.zeros(dm[d].shape, dtype=bool)
    for dd_ in (d, dneg):
        if dd_ in v: mask |= (v[dd_]!=0) | (v[dd_].T!=0)
    m = np.zeros_like(dm[d]); m[mask] = dm[d][mask]
    if d == (0,0,0): np.fill_diagonal(m, np.diag(dm[d]))
    dm_masked[d] = m
mf_sparse = get_mf_normal(v, dm_masked)
err = max(np.max(np.abs(mf_dense[k]-mf_sparse[k])) for k in mf_dense)
scale = max(np.max(np.abs(mf_dense[k])) for k in mf_dense)
print("max|mf(dense dm) - mf(masked dm)| =",err," (mean-field scale",scale,")")
```

## `perf_get_eigenvectors_reshape.py`

Backs: **L6 optimization** — htk.eigenvectors.get_eigenvectors unpacks its already-batched diagonalization with a per-eigenstate Python copy loop

```python
"""LENS6 perf: htk/eigenvectors.get_eigenvectors' dense branch already
batches the diagonalization, but then unpacks it with a per-STATE Python
loop (nk^d * n iterations, one .copy() each, plus an nk^d*n-long Python
list of k-vectors). Show the pure-numpy reshape is BIT-IDENTICAL, and
that the commented-out 'New way' (order='F') in the same file is WRONG."""
import os, tempfile, numpy as np
os.chdir(tempfile.mkdtemp())
from pyqula import geometry
from pyqula.htk.eigenvectors import hk_matrix_batch, parallel_diagonalization
from pyqula.klist import kmesh

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(); h.add_zeeman([0.,0.,0.2]); h.add_rashba(0.15)
nk = 6
f = h.get_hk_gen()
kp = kmesh(h.dimensionality, nk=nk)
mats = hk_matrix_batch(f, kp)
es_batch, ws_batch = parallel_diagonalization(mats)
nkp, n = es_batch.shape
print("nkp =",nkp," n =",n," python iterations in the current loop =",nkp*n)

# ---- current code (htk/eigenvectors.py, "Old way, slightly slower but clearer")
vvs = [(es_batch[i], ws_batch[i]) for i in range(nkp)]
eigvecs = np.zeros((nkp*n, n), dtype=np.complex128)
eigvals = np.zeros(nkp*n)
kvectors = []
iv = 0
for ik in range(nkp):
    vv = vvs[ik]
    for (e, v) in zip(vv[0], vv[1].transpose()):
        eigvecs[iv] = v.copy(); eigvals[iv] = e.copy()
        kvectors.append(kp[ik]); iv += 1

# ---- proposed pure-numpy form
eigvals2 = es_batch.reshape(-1)
eigvecs2 = ws_batch.transpose(0, 2, 1).reshape(nkp*n, n)
kvectors2 = np.repeat(np.array(kp), n, axis=0)

print("max|eigvals diff|  =", np.max(np.abs(eigvals - eigvals2)))
print("max|eigvecs diff|  =", np.max(np.abs(eigvecs - eigvecs2)))
print("bit-identical evals:", np.array_equal(eigvals, eigvals2))
print("bit-identical evecs:", np.array_equal(eigvecs, eigvecs2))
print("max|kvectors diff| =", np.max(np.abs(np.array(kvectors) - kvectors2)))

# ---- the commented-out "New way" in the source (order="F") is wrong
eigvals3 = np.array([iv_[0] for iv_ in vvs]).reshape(nkp*n, order="F")
print("commented-out order='F' form matches? ",
      np.allclose(eigvals, eigvals3))
```

## `perf_kubo_batch2.py`

Backs: **L6 optimization** — conductivitytk/kubo._bands_and_velocities is a serial per-k-point eigh plus a per-k Python rebuild of dH/dk — the last big unbatched k-mesh loop on a public path

```python
"""LENS6 perf: batching conductivitytk/kubo._bands_and_velocities through
hk_matrix_batch+peigh. The intermediate velocity matrix elements are
GAUGE dependent (eigenvector phase differs between scipy's eigh and
numba's numpy eigh), so the oracle must be the gauge-invariant output
(sigma / the Drude weight), not vs itself."""
import os, tempfile, numpy as np
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, conductivity
from pyqula.conductivitytk import kubo
from pyqula.htk.eigenvectors import hk_matrix_batch, parallel_diagonalization

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(); h.add_zeeman([0.,0.,0.2]); h.add_rashba(0.15)
nk=8; T=0.1; delta=0.1; tol=1e-6
ws_ = np.linspace(0.,4.,40)
ks = kubo._kmesh(h,nk)

def batched_bv(h,ks):
    hm,orders,hkgen,jac,dr,cellvol,scale = kubo._setup(h)
    mats = hk_matrix_batch(hkgen,ks)
    es,wsv = parallel_diagonalization(mats)
    n = es.shape[1]; vs = np.zeros((len(ks),3,n,n),dtype=np.complex128)
    for ik,k in enumerate(ks):
        w = wsv[ik]; wc = np.conjugate(w)
        v = kubo._velocities(hm,orders,jac,dr,mats[ik],k)
        for a in range(3): vs[ik,a] = wc.T@v[a]@w
    return es,vs,cellvol,scale

def sigma_from(bv):
    es,vs,cellvol,scale = bv
    ratio,dE = kubo._response_weights(es,T,tol*scale)
    s = kubo._sigma_jit(dE,ratio,vs,ws_,delta)
    return s/(len(ks)*cellvol)

s_cur = sigma_from(kubo._bands_and_velocities(h,ks))
s_new = sigma_from(batched_bv(h,ks))
print("max|sigma_serial - sigma_batched| =", np.max(np.abs(s_cur-s_new)))
print("max|sigma|                        =", np.max(np.abs(s_cur)))
print("max relative difference           =",
      np.max(np.abs(s_cur-s_new))/np.max(np.abs(s_cur)))
```

## `perf_kubo_batch_and_rkky.py`

Backs: **L6 optimization** — rkky_generator recomputes an R-only Bloch-phase array on every (site,site) evaluation, and n-fold redundantly within each one

```python
"""LENS6 perf: (a) conductivitytk/kubo._bands_and_velocities can be
batched through hk_matrix_batch+peigh bit-identically; (b)
chitk/magneticresponse.rkky_generator recomputes an R-only quantity
(bloch_phase over a k-list that repeats every k-point n times) on every
single (ii,jj) evaluation."""
import os, tempfile, numpy as np
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, algebra, rkky
from pyqula.conductivitytk import kubo
from pyqula.htk.eigenvectors import hk_matrix_batch, parallel_diagonalization

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(); h.add_zeeman([0.,0.,0.2]); h.add_rashba(0.15)
ks = kubo._kmesh(h,8)
es,vs,cellvol,scale = kubo._bands_and_velocities(h,ks)
print("(a) _bands_and_velocities: nk =",len(ks)," n =",es.shape[1])
# batched replacement of just the diagonalization half
hm,orders,hkgen,jac,dr,cv,sc = kubo._setup(h)
mats = hk_matrix_batch(hkgen,ks)
es_b,ws_b = parallel_diagonalization(mats)
vs_b = np.zeros_like(vs)
for ik,k in enumerate(ks):
    w = ws_b[ik]; wc = np.conjugate(w)
    v = kubo._velocities(hm,orders,jac,dr,mats[ik],k)
    for a in range(3): vs_b[ik,a] = wc.T@v[a]@w
print("    max|es - es_batched| =", np.max(np.abs(es-es_b)))
print("    max|vs - vs_batched| =", np.max(np.abs(vs-vs_b)))

# (b) rkky_generator redundancy
from pyqula.chitk import magneticresponse as mr
hs = h.copy(); hs.remove_spin()
nk = 8
es2,ws2,ks2 = hs.get_eigenvectors(kpoints=True,nk=nk)
ks2 = np.array(ks2)
nstates = len(es2)
uniq = np.unique(np.round(ks2,12),axis=0)
print("\n(b) rkky_generator(nk=%d, 2d): len(ks) returned by"
      " get_eigenvectors = %d"%(nk,nstates))
print("    distinct k-points in that list        =", len(uniq))
print("    -> bloch_phase is called", nstates, "times per rkky evaluation,")
print("       i.e.", nstates//len(uniq), "x redundantly, and it does not")
print("       depend on (ii,jj) at all, so it is recomputed on every call.")

calls = {"n":0}
orig = hs.geometry.bloch_phase
def counted(d,k):
    calls["n"] += 1; return orig(d,k)
hs.geometry.bloch_phase = counted
gen = mr.rkky_generator(hs,nk=nk)
calls["n"] = 0
for (ii,jj) in [(0,0),(1,0),(0,1)]:
    gen(R=[1,0,0],ii=ii,jj=jj)
print("    measured bloch_phase calls for 3 evaluations at the SAME R:",
      calls["n"])
```

## `perf_multildos_getldosi.py`

Backs: **L6 optimization** — ldos.multi_ldos_tb: a parallel.pcall process pool wrapped around a per-eigenstate Python loop that is one matmul, on top of an unbatched per-k eigh

```python
"""LENS6 perf, HARD RULE 1: ldos.multi_ldos_tb dispatches getldosi over
energies with parallel.pcall (process pool), and getldosi itself is a
pure-Python loop over EVERY eigenstate. The whole thing is one matmul.
Show the matmul is equivalent and count the Python iterations."""
import numpy as np
np.random.seed(0)
nstates, nsite, ne = 6*6*4, 4, 100   # nk=6 2D honeycomb+spin: 144 states
delta = 0.01
evals = np.random.uniform(-3.,3.,nstates)
ws = np.random.normal(size=(nstates,nsite)) + 1j*np.random.normal(size=(nstates,nsite))
ds = [(np.conjugate(v)*v).real for v in ws]
ps = list(np.random.uniform(0.,1.,nstates))
energies = np.linspace(-1.,1.,ne)

# ---- current code (ldos.py multi_ldos_tb.getldosi), once per energy
def getldosi(e):
    out = np.array([0.0 for i in range(nsite)])
    for (d,p,ie) in zip(ds,ps,evals):
        fac = delta/((e-ie)**2 + delta**2)
        out += fac*d*p
    out /= np.pi
    return out
cur = np.array([getldosi(e) for e in energies])
print("python inner iterations now = ne*nstates =", ne*nstates)

# ---- proposed: one (ne,nstates) @ (nstates,nsite) matmul, no pcall
D = np.array(ds)                             # (nstates,nsite)
P = np.array(ps)                             # (nstates,)
W = delta/((energies[:,None]-evals[None,:])**2 + delta**2)*P[None,:]
new = (W @ D)/np.pi
print("max|diff| =", np.max(np.abs(cur-new)))
print("max relative diff =", np.max(np.abs(cur-new))/np.max(np.abs(cur)))
```

## `perf_serial_eigh_counts.py`

Backs: **L6 optimization** — topologytk/qgt._qgt_over_kpoints is a serial per-k-point eigh list comprehension

```python
"""LENS6 perf: count the one-matrix-at-a-time LAPACK calls that remain on
public entry points, i.e. the sites that Tier 1 / its follow-up did NOT
batch through htk.eigenvectors.hk_matrix_batch + peigh/peigvalsh.
No wall-clock numbers: seven other agents are on this machine."""
import os, tempfile, numpy as np
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, algebra, conductivity, topology, ldos
import pyqula.algebra as alg

counts = {}
def wrap(mod,name):
    f = getattr(mod,name); counts[name]=0
    def g(*a,**kw):
        counts[name]+=1; return f(*a,**kw)
    setattr(mod,name,g); return f

orig_eigh  = wrap(alg,"eigh")
orig_eigvh = wrap(alg,"eigvalsh")

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(); h.add_zeeman([0.,0.,0.2]); h.add_rashba(0.2)
n = h.intra.shape[0]
print("matrix dimension n =",n)

for name in counts: counts[name]=0
conductivity.optical_conductivity(h,nk=12,energies=np.linspace(0.,4.,20))
print("optical_conductivity(nk=12) -> algebra.eigh calls:",counts["eigh"],
      " (nk^2 =",12*12,")")

for name in counts: counts[name]=0
conductivity.sum_rule_weight(h,nk=12)
print("sum_rule_weight(nk=12)      -> algebra.eigh calls:",counts["eigh"])

for name in counts: counts[name]=0
topology.quantum_geometric_tensor_mesh(h,nk=12)
print("quantum_geometric_tensor_mesh(nk=12) -> algebra.eigh calls:",counts["eigh"])

for name in counts: counts[name]=0
ldos.multi_ldos(h,nk=6,energies=np.linspace(-1.,1.,10),nrep=1)
print("multi_ldos(nk=6,2d)         -> algebra.eigh calls:",counts["eigh"],
      " (nk^2 =",36,")")
```

## `probe_bcs.py`

Backs: **L7 test coverage** — tests/scf/test_scf_sc_critical_temperature.py pins a gap that is a k-mesh artifact (0.039 at nk=20 vs 0.0055 at nk=200) instead of the nk-robust BCS ratio

```python
"""tests/scf/test_scf_sc_critical_temperature.py pins gap(T) for an attractive
Hubbard chain to recorded constants.  The available published oracle is the BCS
universal ratio Delta(0)/Tc = 1.764 and the flat dDelta/dT -> 0 as T -> 0.
Is the pinned curve consistent with it?"""
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import geometry

def gap(T,nk):
    g = geometry.chain()
    h = g.get_hamiltonian(); h.turn_nambu()
    h = h.get_mean_field_hamiltonian(U=-.6,nk=nk,T=T,mf="random",maxerror=1e-6)
    return h.get_gap()/2.

for nk in (20,200):
    d0 = gap(0.,nk)
    print(f"nk={nk:4d}  Delta(0)={d0:.6f}   BCS Tc = Delta(0)/1.764 = {d0/1.764:.6f}")
    Ts = np.array([0., d0/4, d0/2, 3*d0/4, d0])
    gs = np.array([gap(T,nk) for T in Ts])
    for T,gv in zip(Ts,gs):
        print(f"    T={T:.5f}  (T/Tc_BCS={T/(d0/1.764):.2f})  gap={gv:.6f}  "
              f"gap/Delta(0)={gv/d0:.3f}")
    # what BCS predicts at those T
    print(f"    BCS predicts gap/Delta(0) ~ 1.00, 0.99, 0.93, 0.79, 0.53 "
          f"at T/Tc = 0, 0.44, 0.88, 1.32, 1.76 ... (Tc is at T/Delta0=0.567)")
    print(f"    sum over the test's 3 T's [0, Tmax/2, Tmax] = "
          f"{gs[0]+gs[2]+gs[4]:.6f}  (test pins 0.042088 at nk=20)")
    print()
```

## `probe_oracles.py`

Backs: **L7 test coverage** — tests/topology/test_hall_conductivity.py buries an exactly quantized value (sigma_xy = 2 = Chern number) inside a sum over four nk-sensitive metallic points

```python
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology, ldos

print("=== hall_conductivity: is there a quantized plateau to test against? ===")
for mu in [-0.7,-0.35,0.,0.35,0.7]:
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_zeeman([0.,0.,0.2]); h.add_rashba(0.2); h.shift_fermi(mu)
    s = topology.hall_conductivity(h,nk=8)
    try: c = h.get_chern(nk=14)
    except Exception as e: c = f"ERR {e}"
    print(f"  mu={mu:+.2f}  hall_conductivity={s:+.6f}   get_chern={c}")

print()
print("=== KPM vacancy LDOS vs the exact (ED) LDOS, same system ===")
g = geometry.honeycomb_lattice().supercell(8)
r0=[0.,0.,0.]; g = g.remove(g.closest_index(r0))
h = g.get_hamiltonian(has_spin=False)
E = np.linspace(-1.,1.,60); i = g.closest_index(r0)
es_k,ds_k = ldos.dos_site(h,nk=5,mode="KPM",i=i,energies=E,delta=0.3)
try:
    es_e,ds_e = ldos.dos_site(h,nk=5,mode="ED",i=i,energies=E,delta=0.3)
    print(f"  sum KPM = {np.sum(ds_k):.6f}   sum ED = {np.sum(ds_e):.6f}   "
          f"ratio = {np.sum(ds_k)/np.sum(ds_e):.6f}")
    print(f"  max|KPM-ED| = {np.max(np.abs(np.array(ds_k)-np.array(ds_e))):.4f}, "
          f"max ED = {np.max(ds_e):.4f}")
except Exception as e:
    print("  ED mode:",type(e).__name__,e)
```

## `probe_zero_coverage.py`

Backs: **L7 test coverage** — h.enforce_eh: zero tests, and its NotImplementedError guard is unreachable behind a Python-2 absolute import

Backs: **L7 test coverage** — Zero-coverage public entry points: 16 Hamiltonian/Geometry methods and their delegation targets have no test, no example and no user-guide entry

```python
import numpy as np, traceback
from pyqula import geometry

def probe(name,f):
    try:
        r=f(); print(f"[ok ] {name}: {r}")
    except Exception as e:
        print(f"[ERR] {name}: {type(e).__name__}: {e}")

# --- has_time_reversal_symmetry ---------------------------------------
def trs():
    out={}
    h = geometry.honeycomb_lattice().get_hamiltonian()
    out["plain honeycomb (TRS expected True)"]=h.has_time_reversal_symmetry()
    h2 = h.copy(); h2.add_zeeman([0.,0.,0.3])
    out["Zeeman (expected False)"]=h2.has_time_reversal_symmetry()
    h3 = h.copy(); h3.add_haldane(0.2)
    out["Haldane (expected False)"]=h3.has_time_reversal_symmetry()
    h4 = h.copy(); h4.add_kane_mele(0.2)
    out["Kane-Mele SOC (expected True)"]=h4.has_time_reversal_symmetry()
    h5 = h.copy(); h5.add_rashba(0.2)
    out["Rashba (expected True)"]=h5.has_time_reversal_symmetry()
    return out
probe("has_time_reversal_symmetry",trs)

# --- get_1dh -----------------------------------------------------------
def onedh():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    ky = 0.23
    h1 = h.get_1dh(k=ky)
    hk2 = h.get_hk_gen(); hk1 = h1.get_hk_gen()
    worst=0.
    for kx in [0.,0.1,0.31,0.5,0.77]:
        e2 = np.linalg.eigvalsh(np.array(hk2([kx,ky,0.])))
        e1 = np.linalg.eigvalsh(np.array(hk1([kx,0.,0.])))
        worst=max(worst,np.max(np.abs(np.sort(e1)-np.sort(e2))))
    return f"max |bands(get_1dh) - bands(2D at (kx,ky))| = {worst:.3e}"
probe("get_1dh vs 2D bands",onedh)

# --- fractional round trip --------------------------------------------
def frac():
    g = geometry.honeycomb_lattice().get_supercell(2)
    g.get_fractional()
    r = np.array(g.r)
    g2 = g.copy(); g2.fractional2real()
    d1 = np.max(np.abs(np.array(g2.r)-r))
    g3 = g.copy(); g3.real2fractional(); g3.fractional2real()
    d2 = np.max(np.abs(np.array(g3.r)-r))
    return f"fractional2real drift={d1:.3e}, real2fractional->fractional2real drift={d2:.3e}"
probe("fractional round trip",frac)

# --- same_hamiltonian --------------------------------------------------
def same():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h2 = h.copy(); h3 = h.copy(); h3.add_zeeman([0.,0.,0.9])
    h4 = h.copy(); h4.add_haldane(0.5)
    return (f"same(h,h_copy)={h.same_hamiltonian(h2)}  "
            f"same(h,h+Zeeman)={h.same_hamiltonian(h3)}  "
            f"same(h,h+Haldane)={h.same_hamiltonian(h4)}")
probe("same_hamiltonian",same)

# --- average_spin_splitting -------------------------------------------
def ass():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    a0 = h.get_average_spin_splitting(nk=8)
    h2 = h.copy(); h2.add_zeeman([0.,0.,0.3])
    a1 = h2.get_average_spin_splitting(nk=8)
    return f"no-splitting={a0} (expect 0), Zeeman 0.3 => {a1} (expect ~0.6)"
probe("average_spin_splitting",ass)

# --- to_canonical_gauge ------------------------------------------------
def gauge():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    hk = h.get_hk_gen(); k=[0.3,0.17,0.]
    m = np.array(hk(k))
    mc = np.array(h.to_canonical_gauge(m,k))
    e0=np.sort(np.linalg.eigvalsh(m)); e1=np.sort(np.linalg.eigvalsh(mc))
    return (f"eigenvalue drift under gauge change = {np.max(np.abs(e0-e1)):.3e}; "
            f"hermitian={np.allclose(mc,mc.conj().T)}")
probe("to_canonical_gauge",gauge)
```

## `py2imports.py`

```python
import ast,os
R="<repo root>/src/pyqula"
tops=set()
for f in os.listdir(R):
    if f.endswith(".py"): tops.add(f[:-3])
    elif os.path.isdir(os.path.join(R,f)) and os.path.exists(os.path.join(R,f,"__init__.py")): tops.add(f)
hits=[]
for root,d,fs in os.walk(R):
    if "qutecipytk" in root: continue
    for f in fs:
        if not f.endswith(".py"): continue
        p=os.path.join(root,f)
        try: t=ast.parse(open(p).read())
        except Exception: continue
        for n in ast.walk(t):
            if isinstance(n,ast.ImportFrom) and n.level==0 and n.module and n.module.split(".")[0] in tops:
                hits.append((os.path.relpath(p,R),n.lineno,"from %s import ..."%n.module))
            if isinstance(n,ast.Import):
                for a in n.names:
                    if a.name.split(".")[0] in tops: hits.append((os.path.relpath(p,R),n.lineno,"import "+a.name))
for h in hits: print(h)
print("py2-style absolute imports:",len(hits))
```

## `recorded.py`

```python
import os,re,ast
root="<repo root>/tests"
pat=re.compile(r"-?\d+\.\d{8,}")
rows=[]
for dp,_,fns in os.walk(root):
    for fn in sorted(fns):
        if not fn.startswith("test_") or not fn.endswith(".py"): continue
        p=os.path.join(dp,fn); s=open(p).read()
        t=ast.parse(s)
        tests=[n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name.startswith("test_")]
        nass=sum(len([a for a in ast.walk(n) if isinstance(a,ast.Assert)]) for n in tests)
        lits=pat.findall(s)
        if not lits: continue
        # how many assert statements mention an 8+ digit literal
        rec=0
        for n in tests:
            for a in ast.walk(n):
                if isinstance(a,ast.Assert) and pat.search(ast.unparse(a)): rec+=1
        rows.append((rec/max(nass,1),rec,nass,len(lits),os.path.relpath(p,root)))
rows.sort(reverse=True)
print(f"{'frac':>5} {'rec':>4} {'tot':>4} {'lits':>4}  file")
for r in rows: print(f"{r[0]:5.2f} {r[1]:4d} {r[2]:4d} {r[3]:4d}  {r[4]}")
print("total files with recorded literals:",len(rows))
```

## `refcheck.py`

Backs: **L8 features/docs** — The user guide's 'Main functions and methods' reference omits ten methods the guide's own runnable snippets use

```python

import ast,re,sys
R="<repo root>/"
guide=open(R+"documentation/user_guide.md").read()
lines=guide.split("\n")
# reference section starts at "# Main functions and methods"
for i,l in enumerate(lines):
    if l.strip()=="# Main functions and methods": start=i;break
ref="\n".join(lines[start:])
body="\n".join(lines[:start])
def methods(path,cls):
    t=ast.parse(open(R+path).read())
    for n in ast.walk(t):
        if isinstance(n,ast.ClassDef) and n.name==cls:
            return [f.name for f in n.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))]
    return []
for path,cls in [("src/pyqula/hamiltonians.py","Hamiltonian"),("src/pyqula/geometry.py","Geometry"),
                 ("src/pyqula/heterostructures.py","Heterostructure")]:
    ms=[m for m in methods(path,cls) if not m.startswith("_")]
    missing=[m for m in ms if m not in ref]
    missing_all=[m for m in ms if m not in guide]
    print("### %s: %d public methods, %d absent from reference section, %d absent from the whole guide"%(cls,len(ms),len(missing),len(missing_all)))
    print("  absent from whole guide:", sorted(missing_all))
    print("  in guide body but not reference:", sorted(set(missing)-set(missing_all)))
```

## `refcheck2.py`

```python
import ast,re,sys
R="<repo root>/"
lines=open(R+"documentation/user_guide.md").read().split("\n")
for i,l in enumerate(lines):
    if l.strip()=="# Main functions and methods": start=i;break
def methods(path,cls):
    t=ast.parse(open(R+path).read())
    for n in ast.walk(t):
        if isinstance(n,ast.ClassDef) and n.name==cls:
            return set(f.name for f in n.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)))
    return set()
H=methods("src/pyqula/hamiltonians.py","Hamiltonian")
G=methods("src/pyqula/geometry.py","Geometry")
HT=methods("src/pyqula/heterostructures.py","Heterostructure")
# module-level functions of geometry.py too (factories)
gt=ast.parse(open(R+"src/pyqula/geometry.py").read())
Gmod=set(f.name for f in gt.body if isinstance(f,ast.FunctionDef))
cur=None
bad=[]
for i in range(start,len(lines)):
    l=lines[i].strip()
    if l.startswith("## "): cur=l
    if l.startswith("### "):
        name=l[4:].strip()
        m=re.match(r"^(h|g|ht|sm)\.([A-Za-z_0-9]+)",name)
        if not m: 
            bad.append((i+1,name,"UNPARSED",cur)); continue
        pre,fn=m.group(1),m.group(2)
        pool={"h":H,"g":G|Gmod,"ht":HT,"sm":set()}[pre]
        if fn not in pool: bad.append((i+1,name,"NOT A METHOD of "+pre,cur))
for b in bad: print(b)
print("total suspicious ref entries:",len(bad))
```

## `refcheck3.py`

```python
import ast,re,os
R="<repo root>/"
allnames=set()
for root,d,fs in os.walk(R+"src/pyqula"):
    if "qutecipytk" in root: continue
    for f in fs:
        if not f.endswith(".py"): continue
        try: t=ast.parse(open(os.path.join(root,f)).read())
        except Exception: continue
        for n in ast.walk(t):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): allnames.add(n.name)
            if isinstance(n,ast.Assign):
                for tg in n.targets:
                    if isinstance(tg,ast.Name): allnames.add(tg.id)
lines=open(R+"documentation/user_guide.md").read().split("\n")
for i,l in enumerate(lines):
    if l.strip()=="# Main functions and methods": start=i;break
miss=[]
for i in range(start,len(lines)):
    l=lines[i].strip()
    if l.startswith("### "):
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(",l[4:]):
            nm=m.group(1)
            if nm not in allnames: miss.append((i+1,l[4:],nm))
for x in miss: print(x)
print("missing:",len(miss))
```

## `repro_alias.py`

Backs: **L5 siblings** — h.get_no_multicell() hands back its own input, so mutating the result corrupts the original

Backs: **L5 siblings** — CLEARED: h.get_dense() / h.get_multicell() / h.reduce() aliasing after ccdee4a

```python
import numpy as np, pyqula, io, contextlib
from pyqula import geometry
def bw(h):
    return float(np.max(h.get_bands(nk=4,write=False)[1]))
cases = [
 ("get_dense (already dense)",  lambda h: h.get_dense()),
 ("get_sparse (already sparse)",lambda h: h.get_sparse()),
 ("get_multicell (already mc)", lambda h: h.get_multicell()),
 ("get_no_multicell",           lambda h: h.get_no_multicell()),
 ("reduce (spinless already)",  lambda h: h.reduce()),
 ("copy (control)",             lambda h: h.copy()),
]
for name,f in cases:
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=False)
    if "already sparse" in name: h.turn_sparse()
    b0 = bw(h)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            h2 = f(h)
            h2.add_onsite(1.0)          # mutate the returned object
            b1 = bw(h)                  # did the ORIGINAL change?
    except Exception as e:
        print("%-30s -> %s: %s"%(name,type(e).__name__,str(e)[:70])); continue
    print("%-30s aliased (original corrupted): %s   [%.3f -> %.3f]"
          %(name, abs(b1-b0)>1e-8, b0, b1))
```

## `repro_alloy.py`

Backs: **L5 siblings** — energeticstk.alloytk's default energy path calls exit() and kills the caller's interpreter

```python
import pyqula
from pyqula import geometry
from pyqula.energeticstk.alloytk import Alloy
g = geometry.chain().get_supercell(3)
A = Alloy(g)
print("about to call A.get_energy() ...")
e = A.get_energy()
print("RETURNED", e)     # never reached: exit() kills the interpreter
```

## `repro_attractive_hubbard_mix.py`

Backs: **L1 SCF/Nambu** — attractive_hubbard's convergence test is computed after mixing: the effective tolerance is maxerror/(1-mix), and mix=1.0 "converges" after one iteration

```python
"""FINDING 3 -- scftk/attractive_hubbard_spinless.py:attractive_hubbard computes
its convergence residual AFTER mixing:

    d = f(dold)                            # dold is the current guess
    dold = mix*d + (1-mix)*dold            # <- dold overwritten here
    diff = np.max(np.abs(d-dold))          # <- compared against the NEW dold
    if diff<maxerror: do_scf = False

so diff == (1-mix)*|f(x)-x|.  The effective tolerance is maxerror/(1-mix), and at
mix=1.0 (a perfectly ordinary "no mixing" choice) diff is identically 0, so the
loop stops after ONE iteration from the random guess and returns it as converged.
Every other SCF loop in the package (generic_densitydensity, _run_anisotropic_scf,
generic_densitydensity_kpm) computes the residual BEFORE mixing.
"""
import numpy as np, os, io, contextlib
from pyqula import geometry
from pyqula.scftk.attractive_hubbard_spinless import attractive_hubbard
from pyqula.sctk.spinless import onsite_delta_vev
from pyqula.superconductivity import get_eh_sector
if os.path.exists("MF.pkl"): os.remove("MF.pkl")
G=-2.0; NK=6
for mix in (1.0,0.9,0.5,0.1):
    np.random.seed(0)
    h=geometry.chain().get_hamiltonian(); h.remove_spin()
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        scf=attractive_hubbard(h,g=G,nk=NK,mix=mix,maxerror=1e-6)
    printed=[float(l.split("=")[1]) for l in buf.getvalue().splitlines() if l.startswith("Error = ")]
    delta=np.diag(np.array(get_eh_sector(np.array(scf.hamiltonian.intra),i=0,j=1))) \
          if False else np.array(scf.hamiltonian.intra)[0,1]
    print("mix=%4.2f  iterations=%4d  last TRUE residual printed by f() = %.4e "
          "(tolerance asked for: 1.0e-06)   returned Delta = %+.6f"
          %(mix,len(printed),printed[-1],delta.real))
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
print("(the true fixed point for these parameters is Delta -> 0; the mix=1.0 run "
      "returns a large spurious Delta from a single map application of the random guess)")
```

## `repro_bdg_total_energy.py`

Backs: **L1 SCF/Nambu** — BdG scf.total_energy mixes a doubled band energy with an undoubled double-counting term (not the known anomalous-DC gap — this happens at exactly zero pairing)

```python
"""FINDING 1 -- BdG scf.total_energy is on a different scale from the normal one,
even with EXACTLY ZERO pairing (so this is NOT the documented anomalous
double-counting gap).

spectrum.total_energy (spectrum.py:278/291) sums every eigenvalue below the
Fermi level with no has_eh branch.  For a BdG Hamiltonian
   H = (1/2) Psi^dag H_BdG Psi + (1/2) Tr h
so the electronic energy is (1/2)sum_{E<0}E_BdG + (1/2)Tr h, and summing the
BdG spectrum gives 2*E_electronic - Tr h.  densitydensity.py:549's
`etot += h.fermi*h.intra.shape[0]*filling` uses the Nambu-doubled dimension,
so the muN un-shift is doubled CONSISTENTLY with that -- but the
double-counting term added at line 563 is NOT.  Result: band+muN at 2x scale,
double counting at 1x.
"""
import numpy as np, os
from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector
from pyqula.scftk.densitydensity import get_dc_energy

print("== part A: the band-energy identity, non-interacting ==")
for name,gf in [("chain",geometry.chain),("honeycomb",geometry.honeycomb_lattice)]:
  for mu in (0.0,0.7):
    h = gf().get_hamiltonian(); h.shift_fermi(-mu)
    En = h.get_total_energy(nk=40); tr = np.trace(np.array(h.intra)).real
    hn = gf().get_hamiltonian(); hn.shift_fermi(-mu); hn.setup_nambu_spinor()
    Eb = hn.get_total_energy(nk=40)
    print("  %-10s mu=%.1f E_normal=%10.6f Trh=%8.4f  h.get_total_energy(BdG)=%10.6f  2E-Trh=%10.6f  diff=%.1e"
          %(name,mu,En,tr,Eb,2*En-tr,Eb-(2*En-tr)))

print("\n== part B: SCF with repulsive V1 on a Nambu chain -- zero pairing ==")
g = geometry.chain()
for filling,V1 in [(0.5,1.0),(0.5,2.0),(0.3,1.0)]:
    res={}
    for nambu in (False,True):
        if os.path.exists("MF.pkl"): os.remove("MF.pkl")
        h = g.get_hamiltonian()
        if nambu: h.setup_nambu_spinor()
        h0=h.copy()
        s = meanfield.Vinteraction(h,V1=V1,filling=filling,nk=40,mf="ferroZ",
                mix=0.3,maxerror=1e-9,maxite=2000,load_mf=False,verbose=0)
        mfm = np.array(s.hamiltonian.intra)-np.array(h0.intra)
        pair = np.max(np.abs(get_eh_sector(mfm,i=0,j=1))) if nambu else 0.
        res[nambu]=(s.total_energy,pair)
    print("  filling=%.2f V1=%.1f : normal total_energy=%12.8f  BdG total_energy=%12.8f  diff=%+.8f  (BdG pairing=%.1e)"
          %(filling,V1,res[False][0],res[True][0],res[True][0]-res[False][0],res[True][1]))

print("\n== part C: it is also inconsistent BdG-to-BdG ==")
def tot(nambu,V1):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = g.get_hamiltonian()
    if nambu: h.setup_nambu_spinor()
    return meanfield.Vinteraction(h,V1=V1,filling=0.5,nk=40,mf="ferroZ",mix=0.3,
            maxerror=1e-9,maxite=2000,load_mf=False,verbose=0).total_energy
dn = tot(False,2.0)-tot(False,1.0); db = tot(True,2.0)-tot(True,1.0)
print("  E(V1=2)-E(V1=1):  normal %+.8f   BdG %+.8f   ratio %.4f"%(dn,db,db/dn))
```

## `repro_check_tol.py`

```python
import numpy as np, pyqula
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False); h.get_dense()
h.intra = np.array(h.intra,dtype=complex)
h.intra[0,0] += 5e-5j          # anti-Hermitian defect of size 5e-5
for tol in [1e-6, 1e-4, 1e-3]:
    try:
        h.check(tol=tol); print("tol=%-6g -> did NOT raise"%tol)
    except Exception as e:
        print("tol=%-6g -> %s: %s"%(tol,type(e).__name__,str(e)[:80]))
print("(defect 5e-5: tol=1e-6 SHOULD raise, tol=1e-3 should not;")
print(" check.equal's own default of 1e-4 is what actually decides)")
```

## `repro_check_tol2.py`

Backs: **L5 siblings** — check.check_hermitian's tol never reaches the comparison, so h.check(tol=...) has no effect

```python
import numpy as np, pyqula, io, contextlib
from pyqula import geometry
def trial(defect,tol):
    g = geometry.chain(); h = g.get_hamiltonian(has_spin=False)
    h.intra = np.array(h.intra,dtype=complex); h.intra[0,0] += defect*1j
    try:
        with contextlib.redirect_stdout(io.StringIO()): h.check(tol=tol)
        return "no raise"
    except ValueError: return "RAISED"
for defect in [5e-5, 5e-3]:
    row = [trial(defect,t) for t in [1e-8,1e-6,1e-4,1e-2,1.0]]
    print("anti-Hermitian defect %-8g : tol=1e-8 %-8s 1e-6 %-8s 1e-4 %-8s 1e-2 %-8s 1.0 %s"
          %(defect,*row))
print("the verdict never changes with tol: check.check_hermitian never passes tol to check.equal,")
print("so check.equal's own default 1e-4 is the only threshold that ever applies.")
```

## `repro_constrains_onsite_only.py`

Backs: **L1 SCF/Nambu** — mfconstrains only ever rewrites the onsite (0,0,0) block, so constrains=["no_magnetism"] is a silent no-op for any intersite interaction

```python
"""FINDING 4 -- every routine in scftk/mfconstrains.py only ever rewrites
mf[(0,0,0)] (remove_spinful_sector/remove_spinless_sector: `m = out[(0,0,0)]`),
so with ANY intersite interaction (V1/V2/V3/Vr, or J1/J2/J3 through
VJinteraction/Jinteraction) the spin-dependent Fock mean field that lives on the
BONDS is untouched and constrains=["no_magnetism"] is a silent no-op."""
import numpy as np, os
from pyqula import geometry, meanfield
g = geometry.chain()
for con in ([],["no_magnetism"]):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = g.get_hamiltonian()
    s = meanfield.Vinteraction(h,V1=3.0,filling=0.3,nk=40,mf="ferroZ",
            constrains=con,mix=0.3,maxerror=1e-8,maxite=2000,load_mf=False,verbose=0)
    hc=s.hamiltonian
    b = np.array(s.mf[(1,0,0)])
    print("constrains=%-18s converged=%s"%(str(con),s.converged))
    print("    onsite  mf[(0,0,0)] diag = %s   (no onsite magnetism to remove)"%np.round(np.diag(np.array(s.mf[(0,0,0)])).real,6))
    print("    BOND    mf[(1,0,0)] diag = %s   <- up and down hoppings differ"%np.round(np.diag(b).real,6))
    print("    converged magnetization  = %s"%np.round(hc.get_magnetization(nk=40),6))
```

## `repro_delta_normal.py`

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry,heterostructures
h = geometry.chain().get_hamiltonian(has_spin=False)
def mk(d):
    ht = heterostructures.build(h,h); ht.set_coupling(0.3); ht.delta=d; return ht
E = 1.98   # right at the band edge, where broadening genuinely matters
print("NORMAL (non-BdG) junction, E=%.2f"%E)
print("  ht.delta=1e-6 didv()           =",mk(1e-6).didv(energy=E))
print("  ht.delta=1e-6 didv(delta=3e-1) =",mk(1e-6).didv(energy=E,delta=3e-1))
print("  ht.delta=3e-1 didv()           =",mk(3e-1).didv(energy=E))
```

## `repro_didv_bdg_delta.py`

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry,heterostructures
g = geometry.chain()
h1 = g.get_hamiltonian(has_spin=True); h1.add_swave(0.0)   # normal lead in Nambu
h2 = g.get_hamiltonian(has_spin=True); h2.add_swave(0.1)   # SC lead
ht = heterostructures.build(h1,h2)
ht.delta = 1e-3
print("has_eh:",ht.has_eh)
a = ht.didv(energy=0.05,delta=1e-6)
b = ht.didv(energy=0.05,delta=1e-1)
print("BdG didv(delta=1e-6) =",a)
print("BdG didv(delta=1e-1) =",b)
print("identical:",a==b)
# control: the normal (non-eh) branch does honour delta
gn = geometry.chain()
hn = gn.get_hamiltonian(has_spin=False)
htn = heterostructures.build(hn,hn); htn.delta=1e-3
c = htn.didv(energy=0.3,delta=1e-6); d = htn.didv(energy=0.3,delta=1e-1)
print("normal didv(delta=1e-6) =",c)
print("normal didv(delta=1e-1) =",d)
print("normal identical:",c==d)
```

## `repro_didv_bdg_delta2.py`

Backs: **L8 features/docs** — didv(delta=...) is inert on every scattering-matrix dI/dV path; only the ht.delta attribute matters, and LocalProbe.didv's docstring promises the opposite

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry,heterostructures
g = geometry.chain()
h1 = g.get_hamiltonian(has_spin=True); h1.add_swave(0.0)
h2 = g.get_hamiltonian(has_spin=True); h2.add_swave(0.1)
def mk():
    ht = heterostructures.build(h1,h2); ht.set_coupling(0.3); return ht
E=0.02
ht=mk(); ht.delta=1e-6; A=ht.didv(energy=E)
ht=mk(); ht.delta=1e-6; B=ht.didv(energy=E,delta=2e-1)
ht=mk(); ht.delta=2e-1; C=ht.didv(energy=E)
print("ht.delta=1e-6, didv()            =",A)
print("ht.delta=1e-6, didv(delta=2e-1)  =",B,"  == didv():",B==A)
print("ht.delta=2e-1, didv()            =",C)
print("so didv(delta=) has no effect; the attribute does. ratio C/A =",C/A)

# the LocalProbe case its own docstring advertises
from pyqula.transporttk.localprobe import LocalProbe
hs = geometry.chain().get_hamiltonian(); hs.shift_fermi(1.); hs.add_swave(0.1)
lp1 = LocalProbe(hs,delta=1e-4); lp1.T=0.2
lp2 = LocalProbe(hs,delta=1e-1); lp2.T=0.2
print("LocalProbe(delta=1e-4).didv(E=0.05)            =",lp1.didv(energy=0.05))
print("LocalProbe(delta=1e-4).didv(E=0.05,delta=1e-1) =",lp1.didv(energy=0.05,delta=1e-1))
print("LocalProbe(delta=1e-1).didv(E=0.05)            =",lp2.didv(energy=0.05))
```

## `repro_dos_green_nk.py`

Backs: **L8 features/docs** — h.get_dos(mode="Green"/"RG") ignores nk, and the mode-list in its own ValueError omits both of those accepted modes

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=False)
es = np.linspace(-1.,1.,5)
a = h.get_dos(energies=es,mode="Green",delta=0.1,nk=4)
b = h.get_dos(energies=es,mode="Green",delta=0.1,nk=80)
print("mode=Green nk=4 :",np.round(a[1],8))
print("mode=Green nk=80:",np.round(b[1],8))
print("identical:",np.array_equal(a[1],b[1]))
c = h.get_dos(energies=es,mode="ED",delta=0.1,nk=4)
d = h.get_dos(energies=es,mode="ED",delta=0.1,nk=80)
print("mode=ED nk=4 :",np.round(c[1],6))
print("mode=ED nk=80:",np.round(d[1],6))
print("ED identical:",np.array_equal(c[1],d[1]))
# the error message's accepted list vs what is actually accepted
for m in ["Green","RG","adaptive","ED","KPM","bogus"]:
    try:
        h.get_dos(energies=np.linspace(-1,1,3),mode=m,delta=0.2,nk=4)
        print("mode=%-9s accepted"%m)
    except Exception as ex:
        print("mode=%-9s -> %s: %s"%(m,type(ex).__name__,str(ex)[:120]))
```

## `repro_dos_modes.py`

Backs: **L5 siblings** — dos.get_dos_general's unknown-mode error omits the two modes it actually accepts

```python
import numpy as np, pyqula, os
os.chdir("SCRATCH")
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False)   # 1 orbital -> integral of DOS must be 1
es = np.linspace(-3.,3.,600)
res={}
for mode in ["ED","Green","adaptive"]:
    try:
        x,y = h.get_dos(energies=es,mode=mode,delta=2e-2,nk=400)
        res[mode]=(np.array(x),np.array(y))
        print("%-9s integral = %.5f   max=%.4f"%(mode,np.trapezoid(y,x),np.max(y)))
    except Exception as e:
        print("%-9s FAILED %s: %s"%(mode,type(e).__name__,str(e)[:120]))
try:
    x,y = h.get_dos(energies=es,mode="KPM",delta=2e-2,nk=400)
    print("%-9s integral = %.5f   max=%.4f"%("KPM",np.trapezoid(y,x),np.max(y)))
    res["KPM"]=(np.array(x),np.array(y))
except Exception as e:
    print("KPM FAILED",type(e).__name__,str(e)[:160])
```

## `repro_dos_pi.py`

Backs: **L5 siblings** — h.get_dos(mode="adaptive") returns a DOS exactly pi times too large

Backs: **L5 siblings** — dos_ewindow's DOS is pi too large, and dos1d_ewindow's use_green is dead while the 2d sibling honours it

```python
import numpy as np, pyqula, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from pyqula import geometry, dos
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False)      # 1 state per cell -> int DOS dE == 1
es = np.linspace(-3.,3.,601)

print("### A. h.get_dos(mode=...) : the public entry point ###")
for mode in ["ED","adaptive"]:
    x,y = h.get_dos(energies=es,mode=mode,delta=2e-2,nk=400)
    print("   mode=%-9s  integral over E = %.5f  (must be 1.0)"%(mode,np.trapezoid(y,x)))
xED,yED = h.get_dos(energies=es,mode="ED",delta=2e-2,nk=400)
xAD,yAD = h.get_dos(energies=es,mode="adaptive",delta=2e-2,nk=400)
print("   adaptive/ED ratio (median) = %.6f ;  pi = %.6f"%(np.median(np.array(yAD)/np.array(yED)),np.pi))

print("### B. dos.dos_ewindow vs dos.dos_kmesh, same system ###")
dos.dos_ewindow(h,energies=es,delta=2e-2,nk=400,use_green=False)
ew = np.genfromtxt("DOS.OUT").T
print("   dos_ewindow integral = %.5f  (must be 1.0)"%np.trapezoid(ew[1],ew[0]))
print("   dos_ewindow/ED ratio (median) = %.6f"%np.median(ew[1]/np.array(yED)))

print("### C. dos1d_ewindow ignores use_green (dos2d_ewindow honours it) ###")
dos.dos1d_ewindow(h,energies=es,delta=2e-2,nk=400,use_green=True)
a = np.genfromtxt("DOS.OUT").T
dos.dos1d_ewindow(h,energies=es,delta=2e-2,nk=400,use_green=False)
b = np.genfromtxt("DOS.OUT").T
print("   use_green=True vs False byte-identical:",np.array_equal(a,b))
```

## `repro_embed2.py`

Backs: **L5 siblings** — CLEARED: embeddingtk/ldos.py:35 applies operator*G one-sided instead of ldos.green2ldos's Hermitian (GA+AG)/2

```python
import numpy as np, pyqula
from pyqula import geometry, embedding
from pyqula.ldos import green2ldos
from pyqula.increase_hilbert import full2profile
np.random.seed(0)
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_zeeman([0.3,0.,0.])      # REAL Hamiltonian, spin-mixing, sy identically zero
print("Hamiltonian purely real?", np.max(np.abs(np.array(h.intra).imag))==0.)
hd = h.copy(); hd.add_onsite(lambda r: 0.5 if np.sum(r**2)<1e-3 else 0.)
EB = embedding.Embedding(h, m=hd.intra)
for op in ["sy","sx","sz"]:
    x,y,d = EB.get_ldos(energy=0.15,delta=1e-2,nsuper=1,nk=20,operator=op,write=False)
    print("Embedding.get_ldos operator=%s  max|.| = %.6g"%(op,np.max(np.abs(d))))
gv = np.array(EB.get_gf(energy=0.15,delta=1e-2,nsuper=1,nk=20))
for op in ["sy","sx","sz"]:
    A = h.get_operator(op).get_matrix()
    one = full2profile(h,-np.diag(A@gv).imag/np.pi,check=False)
    her = full2profile(h,green2ldos(gv,op=A),check=False)
    print("  %s: one-sided(AG) max=%.6g   hermitian max=%.6g"%(op,np.max(np.abs(one)),np.max(np.abs(her))))
print("--- for reference, the FIXED bulk path ldos.get_ldos(mode='green') ---")
for op in ["sy","sx"]:
    out = h.get_ldos(e=0.15,delta=1e-2,mode="green",operator=op,nrep=1,write=False,nk=20)
    print("  h.get_ldos operator=%s max|.| = %.6g"%(op,np.max(np.abs(out[-1]))))
```

## `repro_embed3.py`

Backs: **L5 siblings** — CLEARED: embeddingtk/ldos.py:35 applies operator*G one-sided instead of ldos.green2ldos's Hermitian (GA+AG)/2

```python
import numpy as np, pyqula
from pyqula import geometry, embedding
from pyqula.ldos import green2ldos
from pyqula.increase_hilbert import full2profile
np.random.seed(0)
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_rashba(0.3); h.add_zeeman([0.1,0.2,0.3])   # complex Hamiltonian
hd = h.copy(); hd.add_onsite(lambda r: 0.5 if np.sum(r**2)<1e-3 else 0.)
EB = embedding.Embedding(h, m=hd.intra)
gv = np.array(EB.get_gf(energy=0.15,delta=1e-2,nsuper=1,nk=20))
for op in ["sx","sy","sz"]:
    A = h.get_operator(op).get_matrix()
    one = full2profile(h,-np.diag(A@gv).imag/np.pi,check=False)
    her = full2profile(h,green2ldos(gv,op=A),check=False)
    print("%s: one-sided=%s"%(op,np.round(one,6)))
    print("%s: hermitian=%s"%(op,np.round(her,6)))
    print("   max abs diff = %.6g   (relative %.3g)"%(np.max(np.abs(one-her)),
          np.max(np.abs(one-her))/max(np.max(np.abs(her)),1e-30)))
```

## `repro_embed_ldos_op.py`

```python
import numpy as np, pyqula
assert "/src/pyqula" in pyqula.__file__
from pyqula import geometry, embedding
np.random.seed(0)

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)       # purely REAL Hamiltonian, no SOC
h.add_zeeman([0.,0.,0.3])                  # polarized along z only -> sy == 0 everywhere
hd = h.copy(); hd.add_onsite(lambda r: 0.5 if np.sum(r**2)<1e-3 else 0.) # a defect
EB = embedding.Embedding(h, m=hd.intra)

x,y,dsy = EB.get_ldos(energy=0.15, delta=1e-2, nsuper=1, nk=20,
                      operator="sy", write=False)
x,y,dsz = EB.get_ldos(energy=0.15, delta=1e-2, nsuper=1, nk=20,
                      operator="sz", write=False)
x,y,d0  = EB.get_ldos(energy=0.15, delta=1e-2, nsuper=1, nk=20, write=False)
print("embedding sy-LDOS  max|.| =", np.max(np.abs(dsy)))
print("embedding sz-LDOS  max|.| =", np.max(np.abs(dsz)))
print("embedding  LDOS    max|.| =", np.max(np.abs(d0)))
print("ratio |sy|/|ldos|  =", np.max(np.abs(dsy))/np.max(np.abs(d0)))

# oracle: the same contraction done the Hermitian way, as ldos.green2ldos does
from pyqula.ldos import green2ldos
from pyqula.increase_hilbert import full2profile
gv = EB.get_gf(energy=0.15, delta=1e-2, nsuper=1, nk=20)
op_sy = h.get_operator("sy").get_matrix()
op_sz = h.get_operator("sz").get_matrix()
for nm,op in [("sy",op_sy),("sz",op_sz)]:
    one_sided = -np.diag(op@np.array(gv)).imag/np.pi
    herm      = green2ldos(gv,op=op)
    print(nm, " one-sided max|.| =", np.max(np.abs(full2profile(h,one_sided,check=False))),
              " hermitian max|.| =", np.max(np.abs(full2profile(h,herm,check=False))))
```

## `repro_embedding_gf_op.py`

Backs: **L8 features/docs** — Embedding.get_gf accepts operator= and ignores it, while Embedding.get_ldos in the same class honours it

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry, embedding
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)
h.add_exchange([0.,0.,0.5])
hv = h.copy(); hv.add_onsite(lambda r: 3.0 if np.abs(r[0])<1e-6 else 0.0)
eb = embedding.Embedding(h,m=hv.intra)
a = eb.get_gf(energy=0.2,delta=0.1,nk=6)
b = eb.get_gf(energy=0.2,delta=0.1,nk=6,operator="sz")
print("get_gf(operator=None) trace:",np.round(np.trace(a),8))
print("get_gf(operator='sz')  trace:",np.round(np.trace(b),8))
print("byte identical:",np.array_equal(a,b))
# the sibling in the same class that DOES honour operator
l0 = eb.get_ldos(energy=0.2,delta=0.1,nk=6)
l1 = eb.get_ldos(energy=0.2,delta=0.1,nk=6,operator="sz")
d0 = np.array(l0[-1]); d1 = np.array(l1[-1])
print("get_ldos None sum:",np.round(d0.sum(),6)," sz sum:",np.round(d1.sum(),6),
      " identical:",np.array_equal(d0,d1))
```

## `repro_enforce_eh.py`

Backs: **L8 features/docs** — h.enforce_eh() raises ModuleNotFoundError from a Python-2 absolute import placed above its own NotImplementedError

```python
import sys
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry
h = geometry.chain().get_hamiltonian()
h.setup_nambu_spinor()
try: h.enforce_eh()
except Exception as e: print("h.enforce_eh() ->",type(e).__name__+":",e)
```

## `repro_finiteT_filling.py`

Backs: **L1 SCF/Nambu** — Finite-temperature SCF at fixed filling silently runs at the wrong electron count: the Fermi level comes from a T=0 eigenvalue count

```python
import numpy as np, os
from pyqula import geometry, meanfield
def ne(s): return np.trace(np.array(s.dm[(0,0,0)])).real
g=geometry.chain()
for T in (1e-7,1e-3,0.01,0.05):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h=g.get_hamiltonian()
    s=meanfield.Vinteraction(h,U=1.0,filling=0.1,T=T,nk=60,mf="ferroZ",mix=0.3,
        maxerror=1e-7,maxite=400,load_mf=False,verbose=0)
    print("filling=0.10 T=%-8.4g requested N=0.200000  converged N=%.6f  err=%+.2f%%"%(T,ne(s),100*(ne(s)-0.2)/0.2))
# and away from a particle-hole symmetric point, for a range of fillings
print()
for filling,T in [(0.1,0.05),(0.2,0.1),(0.3,0.2),(0.3,0.5)]:
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h=g.get_hamiltonian()
    s=meanfield.Vinteraction(h,U=1.0,filling=filling,T=T,nk=60,mf="ferroZ",mix=0.3,
        maxerror=1e-7,maxite=400,load_mf=False,verbose=0)
    print("filling=%.2f T=%-6.3g requested N=%.6f  converged N=%.6f  err=%+.2f%%"%(
        filling,T,2*filling,ne(s),100*(ne(s)-2*filling)/(2*filling)))
```

## `repro_frand.py`

Backs: **L5 siblings** — kdos.kdos_bands accepts frand and never calls it, in either mode

```python
import numpy as np, pyqula, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from pyqula import geometry, kdos
g = geometry.honeycomb_zigzag_ribbon(15)
h = g.get_hamiltonian(); h.add_haldane(0.1)
edge = np.zeros(h.intra.shape[0]); edge += 1.0; edge[10:] = 0.0
calls = {"n":0}
def frand():
    calls["n"] += 1
    return (-0.5+np.random.random(edge.shape[0]))*edge
es = np.linspace(-3.,3.,40)
for mode in ["ED","KPM"]:
    calls["n"]=0
    np.random.seed(1)
    a = kdos.kdos_bands(h,frand=frand,nk=5,energies=es,mode=mode,ntries=3)
    n1 = calls["n"]
    np.random.seed(1)
    b = kdos.kdos_bands(h,nk=5,energies=es,mode=mode,ntries=3)
    print("mode=%-4s frand called %d times;  with/without frand identical: %s"
          %(mode,n1,np.array_equal(a,b)))
```

## `repro_guide_numbers.py`

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry
# user_guide.md L3270-3273
h = geometry.chain().get_hamiltonian()
h.add_exchange([0.,0.,0.5])
print("guide says [0,0,-0.15]:",np.round(h.get_magnetization(nk=40),4))
print("guide says [0,0,0.5]  :",np.round(h.get_magnetization(mode="field"),4))
```

## `repro_holes.py`

Backs: **L5 siblings** — specialhamiltonian.SOC_TMDC accepts soc and hardcodes phi=0.5

Backs: **L5 siblings** — dos.surface_dos accepts operator and never uses it

Backs: **L5 siblings** — embedding.get_gf_exact accepts operator and returns the unprojected Green's function

Backs: **L5 siblings** — fermisurface.fermi_surface_generator accepts refine_delta and never uses it

```python
import numpy as np, pyqula, os, io, contextlib
os.chdir("SCRATCH")
from pyqula import geometry, specialhamiltonian, dos, embedding, fermisurface, bandstructure, ldos
q=lambda f,*a,**k: f(*a,**k)
with contextlib.redirect_stdout(io.StringIO()):
    A = specialhamiltonian.SOC_TMDC(soc=0.0); B = specialhamiltonian.SOC_TMDC(soc=0.9)
from pyqula.algebra import todense
same = np.array_equal(todense(A.intra),todense(B.intra)) and all(
    np.array_equal(todense(a.m),todense(b.m)) for a,b in zip(A.hopping,B.hopping))
print("SOC_TMDC(soc=0.0) == SOC_TMDC(soc=0.9):",same)

g=geometry.chain(); h=g.get_hamiltonian(has_spin=True); h.add_zeeman([0,0,0.4]); h=h.get_no_multicell()
with contextlib.redirect_stdout(io.StringIO()):
    e1,d1 = dos.surface_dos(h,energies=np.linspace(-1,1,8),delta=0.1)
    e2,d2 = dos.surface_dos(h,energies=np.linspace(-1,1,8),delta=0.1,operator="sz")
print("dos.surface_dos(operator='sz') identical to no operator:",np.array_equal(d1,d2))

g2=geometry.honeycomb_lattice(); h2=g2.get_hamiltonian(has_spin=True); h2.add_zeeman([0,0,0.4])
EB=embedding.Embedding(h2,m=h2.intra)
a=np.array(EB.get_gf(energy=0.2,delta=1e-2,nk=8))
b=np.array(EB.get_gf(energy=0.2,delta=1e-2,nk=8,operator="sz"))
print("Embedding.get_gf(operator='sz') identical to no operator:",np.array_equal(a,b))

with contextlib.redirect_stdout(io.StringIO()):
    f1=fermisurface.fermi_surface_generator(h2,nk=8,refine_delta=1.0)
    f2=fermisurface.fermi_surface_generator(h2,nk=8,refine_delta=50.0)
print("fermi_surface_generator refine_delta=1 vs 50 identical:",np.array_equal(np.array(f1[-1]),np.array(f2[-1])))

hc=geometry.chain().get_supercell(4).get_hamiltonian(has_spin=False)
with contextlib.redirect_stdout(io.StringIO()):
    l1=bandstructure.lowest_bands(hc,nkpoints=10,nbands=2)
    l2=bandstructure.lowest_bands(hc,nkpoints=200,nbands=2)
print("lowest_bands nkpoints=10 vs 200 identical:",np.array_equal(np.array(l1),np.array(l2)))
print("ldos.ldos_potential(h) returns:",ldos.ldos_potential(h2))
```

## `repro_holes2.py`

```python
import numpy as np, pyqula, os, io, contextlib
os.chdir("SCRATCH")
from pyqula import geometry, bandstructure, ldos, klist
g = geometry.honeycomb_zigzag_ribbon(6)
h = g.get_hamiltonian(has_spin=False)
with contextlib.redirect_stdout(io.StringIO()):
    l1 = bandstructure.lowest_bands(h,nkpoints=10,nbands=4)
    l2 = bandstructure.lowest_bands(h,nkpoints=300,nbands=4)
print("lowest_bands: len(nkpoints=10) =",len(np.atleast_1d(np.genfromtxt("BANDS.OUT"))))
print("lowest_bands nkpoints=10 vs 300 give identical BANDS.OUT:",np.array_equal(np.array(l1),np.array(l2)))
print("   klist.default path length (what it actually uses):",len(klist.default(h.geometry)))
print("ldos.ldos_potential(h) returns:",ldos.ldos_potential(h))
```

## `repro_kondo_hs_constant.py`

Backs: **L1 SCF/Nambu** — Kondo-lattice total energy is missing the factor N=2 on the Hubbard-Stratonovich constant, so the mean-field energy is not stationary at the SCF's own fixed point

```python
"""FINDING 2 -- scftk/kondolattice.py:_pack's Hubbard-Stratonovich constant is
  hs_term = sum |V|^2 / J
but must be  N |V|^2 / J = 2 |V|^2 / J  for N=2.

Derivation (independent of Coleman's equation numbering):
  H_I = (J/N) S_ab c^dag_b c_a  with S_ab=f^dag_a f_b   =>  H_I = -(J/N) X^dag X,
  X = sum_b c^dag_b f_b.  Decoupling -g X^dag X (g=J/N) as  Vbar X + X^dag V + |V|^2/g
  is stationary at V=-g X, and gives back -g|X|^2 only with the constant |V|^2/g
  = N|V|^2/J.  The code's own update, V <- -(J/2)*A, already uses g=J/2=J/N, so it
  is the constant that is inconsistent with it, not the update.

Independent oracle: Hellmann-Feynman.  Omega(V) = sum_occ(e-mu) + c|V|^2/J - lam*Q
must be STATIONARY in V at the SCF's own fixed point.
  dOmega/dV = A + c*Vbar/J ,  and the fixed point has A = -2V/J,
so it vanishes only for c=2.
"""
import numpy as np
from pyqula import geometry
from pyqula.kondolattice import KondoLatticeHamiltonian
from pyqula.multihopping import MultiHopping
J=1.5 ; filling=0.15 ; nk=200 ; T=2e-2 ; Q=1.0 ; Jg=J/2.
gc=geometry.chain(); K=KondoLatticeHamiltonian(gc.get_hamiltonian(has_spin=True))
pairs=K._kondo_pairs ; h1=K.get_dense() ; hop0=h1.get_dict()
mu=h1.get_fermi4filling(filling,nk=nk)
def build_extra(V,lam):
    m=np.zeros(h1.intra.shape,dtype=np.complex128)
    for idx,(ci,fi) in enumerate(pairs):
        for s in (0,1):
            cc,ff=2*ci+s,2*fi+s
            m[ff,ff]+=lam[idx]; m[cc,ff]+=np.conjugate(V[idx]); m[ff,cc]+=V[idx]
    return {(0,0,0):m}
def omega(V,lam,c):
    h=h1.copy(); h.set_multihopping(MultiHopping(hop0)+MultiHopping(build_extra(V,lam)))
    h.fermi=mu; h.shift_fermi(-mu)
    dm=h.get_density_matrix(nk=nk,T=T,ds=[(0,0,0)])[(0,0,0)]
    A=np.array([dm[2*fi,2*ci]+dm[2*fi+1,2*ci+1] for (ci,fi) in pairs])
    return h.get_total_energy(nk=nk)+c*np.sum(np.abs(V)**2)/J-np.sum(lam)*Q, A
h2=K.get_mean_field_hamiltonian(J=J,filling=filling,nk=nk,
        mf=(np.array([0.3+0j]),np.array([0.0])),mix=0.3,maxerror=1e-8,maxite=5000)
V=h2.hybridization.copy(); lam=h2.constraint_lambda.copy()
O,A=omega(V,lam,1.0)
print("SCF fixed point  V*=%.8f  lam=%.8f  n_f=%.6f"%(V[0].real,lam[0],h2.local_occupation[0]))
print("  A=<f^dag c>=%.8f ;  -J/2*A=%.8f == V*  (the code's own self-consistency)"%(A[0].real,(-Jg*A[0]).real))
print("  dOmega/dV, code's c=1 : %+.8f   <-- NOT stationary"%((A[0]+1*V[0]/J).real))
print("  dOmega/dV, c=N=2      : %+.3e   <-- stationary"%((A[0]+2*V[0]/J).real))
# impact
from pyqula.kondolattice import KondoLatticeHamiltonian as KLH
def mk(): return KLH(geometry.chain().get_hamiltonian(has_spin=True))
hK,eK=mk().get_mean_field_hamiltonian(J=J,filling=filling,nk=nk,
        mf=(np.array([0.3+0j]),np.array([0.0])),mix=0.3,maxerror=1e-8,maxite=5000,
        return_total_energy=True)
hT,eT=mk().get_mean_field_hamiltonian(J=J,filling=filling,nk=nk,mix=0.3,
        maxerror=1e-8,maxite=500,return_total_energy=True)
corr=np.sum(np.abs(hK.hybridization)**2)/J
print("  reported  E_kondo=%.8f  E_trivial=%.8f  dE=%.8f"%(eK,eT,eK-eT))
print("  corrected E_kondo=%.8f  E_trivial=%.8f  dE=%.8f   (condensation energy overstated 2.76x)"
      %(eK+corr,eT,eK+corr-eT))
```

## `repro_kys_ignored.py`

Backs: **L6 optimization** — The 2D k-map loops iterate kxs for the y axis and silently discard kys, so k0[1] is ignored

```python
"""LENS6: the batched 2D k-map loops build their k-point list as
  xys = [(x,y) for x in kxs for y in kxs]
i.e. they iterate the *kx* array for the y axis and never use kys.
Oracle: the y column of the written output must carry k0[1]'s offset."""
import os, tempfile, numpy as np
d = tempfile.mkdtemp(); os.chdir(d)
from pyqula import geometry, spectrum, spintexture

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(); h.add_zeeman([0.,0.,0.3])

k0 = [0.0, 0.5]     # asymmetric shift: kxs centred on 0, kys centred on 0.5
nk = 3
spectrum.ev2d(h, nk=nk, k0=k0)
m = np.genfromtxt("EV2D.OUT")
print("ev2d(k0=[0.0,0.5], nk=3, nsuper=1)")
print("  kx column :", np.unique(np.round(m[:,0],6)))
print("  ky column :", np.unique(np.round(m[:,1],6)))
print("  expected ky (=linspace(-1,1,3)+0.5):",
      np.round(np.linspace(-1.,1.,3)+k0[1],6))

spectrum.selected_bands2d(h, nindex=[1], nk=nk, k0=k0, output_file="BANDS2D_")
m2 = np.genfromtxt("BANDS2D__1.OUT")
print("selected_bands2d(k0=[0.0,0.5])")
print("  ky column :", np.unique(np.round(m2[:,1],6)))

# spintexture.kfun_map: same "for y in kxs"
kx,ky,out = spintexture.kfun_map(h, nk=3, k0=k0,
        operator=lambda hk: np.trace(hk).real)
print("kfun_map(k0=[0.0,0.5])")
print("  ky values :", np.unique(np.round(ky,6)))
```

## `repro_ldos_op.py`

Backs: **L5 siblings** — h.get_operator("ldos") builds a matrix of the wrong dimension and is unusable on any periodic Hamiltonian

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry, operatorlist
for label,h in [("spinless chain",geometry.chain().get_hamiltonian(has_spin=False)),
                ("spinful honeycomb",geometry.honeycomb_lattice().get_hamiltonian()),
                ("spinless honeycomb x4",geometry.honeycomb_lattice().get_supercell(2).get_hamiltonian(has_spin=False))]:
    try:
        o=h.get_bands(nk=3,operator="ldos"); print("%-24s OK"%label)
    except Exception as e: print("%-24s %s: %s"%(label,type(e).__name__,str(e)[:100]))
import inspect
from pyqula import operators
print(inspect.getsource(operatorlist.__dict__["operator_dict"]) if "operator_dict" in operatorlist.__dict__ else "")
```

## `repro_ldos_op2.py`

Backs: **L8 features/docs** — h.get_operator("ldos") is broken for every periodic Hamiltonian - the projector is built on the nrep-replicated real-space grid, not the unit cell

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry, ldos
# 0d island
g = geometry.honeycomb_lattice().get_supercell(3)
g = g.supercell(1) if False else g
g.dimensionality = 0
h = g.get_hamiltonian(has_spin=False)
print("H dim:",h.intra.shape)
try:
    op = ldos.ldos_projector(h,e=0.0)
    print("0d ldos_projector matrix shape:",op.matrix.shape)
except Exception as e: print("0d:",type(e).__name__,str(e)[:120])
# periodic
h2 = geometry.chain().get_hamiltonian(has_spin=False)
op2 = ldos.ldos_projector(h2,e=0.0)
print("1d chain: H dim",h2.intra.shape," ldos operator matrix shape",op2.matrix.shape)
try:
    h2.get_bands(nk=3,operator="ldos")
except Exception as e: print("  get_bands(operator='ldos'):",type(e).__name__,str(e)[:100])
```

## `repro_ldos_oracle.py`

Backs: **L5 siblings** — h.get_multildos LDOS maps are nk^dim too large and its DOS.OUT is pi*nk^dim too large

Backs: **L5 siblings** — CLEARED: h.get_ldos mode="arpack" vs mode="green" after 05e0f17 made both honour operator=

```python
import numpy as np, pyqula, os
os.chdir("SCRATCH")
from pyqula import geometry, klist
g = geometry.honeycomb_lattice(); h = g.get_hamiltonian(has_spin=False)
e=0.4; delta=0.05; nk=30
ks = klist.kmesh(h.dimensionality,nk=nk); hk=h.get_hk_gen()
acc=np.zeros(2)
for k in ks:
    ev,w = np.linalg.eigh(np.array(hk(k)))
    for i in range(len(ev)):
        acc += (delta/((e-ev[i])**2+delta**2))*np.abs(w[:,i])**2
acc = acc/np.pi/len(ks)
print("oracle  per-site LDOS      :",np.round(acc,6))
out = h.get_ldos(e=e,delta=delta,mode="arpack",nk=nk,nrep=1,write=False)
print("h.get_ldos per-site LDOS   :",np.round(out[-1],6))
d=np.atleast_2d(np.genfromtxt("MULTILDOS/LDOS_"+str(e)+"_.OUT"))
print("h.get_multildos per-site   :",np.round(d[:,2],6),"  (ratio %.1f)"%(d[0,2]/acc[0]))
```

## `repro_ldospot.py`

Backs: **L5 siblings** — ldos.ldos_potential silently returns None

```python
import pyqula
from pyqula import geometry, ldos
h = geometry.honeycomb_lattice().get_hamiltonian()
print("ldos.ldos_potential(h) returns:", repr(ldos.ldos_potential(h)))
```

## `repro_leftovers.py`

Backs: **L8 features/docs** — Nine public functions are dead on arrival with undefined names or an unimportable module

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry
def t(label,f):
    try: f(); print("  %-45s OK"%label)
    except Exception as e: print("  %-45s %s: %s"%(label,type(e).__name__,str(e)[:90]))
g=geometry.chain(); h=g.get_hamiltonian()
print("--- sctk.pairing mode='deltaud' (a listed elif branch)")
hn=g.get_hamiltonian(); hn.setup_nambu_spinor()
t("h.add_pairing(mode='deltaud',delta=0.1)",lambda: hn.add_pairing(delta=0.1,mode="deltaud"))
hn2=g.get_hamiltonian(); hn2.setup_nambu_spinor()
t("h.add_pairing(mode='swave',delta=0.1) control",lambda: hn2.add_pairing(delta=0.1,mode="swave"))
print("--- pyqula.current")
from pyqula import current
t("current.gs_current(h,nk=4)",lambda: current.gs_current(h,nk=4))
print("--- pyqula.sculpt.build_ribbon")
from pyqula import sculpt
g2=geometry.square_lattice()
t("sculpt.build_ribbon(square,3)",lambda: sculpt.build_ribbon(g2,3))
print("--- pyqula.green.full_inverse")
from pyqula import green
m=[[np.identity(2)*2,np.identity(2)],[np.identity(2),np.identity(2)*2]]
t("green.full_inverse(m)",lambda: green.full_inverse(m))
print("--- pyqula.algebra.spectral_gap fallback (all eigenvalues one sign)")
from pyqula import algebra
mm=np.diag([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.])
t("algebra.spectral_gap(all-positive matrix)",lambda: algebra.spectral_gap(mm))
print("--- pyqula.spectrum.ev2d on a sparse 2d Hamiltonian")
from pyqula import spectrum
h2=geometry.honeycomb_lattice().get_hamiltonian(has_spin=False); h2.turn_sparse()
t("spectrum.ev2d(sparse 2d,nk=2)",lambda: spectrum.ev2d(h2,nk=2))
print("--- pyqula.inout.writefile decorator")
from pyqula import inout
t("inout.writefile-decorated function",lambda: inout.writefile(None)(lambda: [1,2])())
print("--- heterostructures.plot_central_dos")
from pyqula import heterostructures as HS
h1=g.get_hamiltonian()
ht=HS.build(h1,h1)
t("HS.plot_central_dos(ht)",lambda: HS.plot_central_dos(ht))
print("--- scftk submodule import order")
t("import pyqula.scftk.hubbard",lambda: __import__("pyqula.scftk.hubbard"))
```

## `repro_localprobe_T.py`

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe
g = geometry.chain(); h = g.get_hamiltonian(has_spin=False)
lp = LocalProbe(h,delta=1e-3)
lp.T = 0.2
base = lp.didv(energy=0.3)
passed = lp.didv(energy=0.3,T=0.9)   # ask for a different transparency
lp.T = 0.9
attr  = lp.didv(energy=0.3)          # the same thing set as an attribute
print("lp.T=0.2, didv()          =",base)
print("lp.T=0.2, didv(T=0.9)     =",passed, " identical to base:",passed==base)
print("lp.T=0.9, didv()          =",attr)
print("ratio attr/base           =",attr/base)
```

## `repro_localprobe_T2.py`

Backs: **L8 features/docs** — LocalProbe.didv(T=...) swallows the temperature alias that Heterostructure.didv honours and that a test pins

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry,heterostructures
from pyqula.transporttk.localprobe import LocalProbe
h = geometry.chain().get_hamiltonian(has_spin=False)
E=1.9
print("Heterostructure (T is the temperature alias, pinned by tests/scf/test_silently_ignored_inputs.py:99)")
ht=heterostructures.build(h,h); ht.set_coupling(0.3); ht.delta=1e-3
a=ht.didv(energy=E); b=ht.didv(energy=E,T=0.2); c=ht.didv(energy=E,temp=0.2)
print("  didv()      =",a)
print("  didv(T=0.2) =",b,"  differs from didv():",b!=a)
print("  didv(temp=0.2)=",c,"  T==temp:",abs(b-c)<1e-12)
print("LocalProbe, same three calls")
lp=LocalProbe(h,delta=1e-3); lp.T=0.3
a=lp.didv(energy=E); b=lp.didv(energy=E,T=0.2); c=lp.didv(energy=E,temp=0.2)
print("  didv()      =",a)
print("  didv(T=0.2) =",b,"  IDENTICAL to didv():",b==a)
print("  didv(temp=0.2)=",c,"  differs from didv():",c!=a)
```

## `repro_lowest.py`

Backs: **L5 siblings** — bandstructure.lowest_bands accepts nkpoints and always uses klist.default's fixed path

```python
import numpy as np, pyqula, os, io, contextlib, inspect
os.chdir("SCRATCH")
from pyqula import geometry, bandstructure, klist
g = geometry.honeycomb_zigzag_ribbon(4)
h = g.get_hamiltonian(has_spin=False)
kp = klist.default(h.geometry)
print("klist.default path length used regardless of nkpoints:",len(kp))
src = inspect.getsource(bandstructure.lowest_bands)
print("nkpoints appears in the body:", "nkpoints" in src.split("\n",1)[1].split('"""')[-1])
```

## `repro_lowest2.py`

```python
import numpy as np, pyqula, os, io, contextlib
os.chdir("SCRATCH")
from pyqula import geometry, bandstructure
g = geometry.honeycomb_zigzag_ribbon(3)
h = g.get_hamiltonian(has_spin=False)
outs=[]
for nkp in [8,500]:
    with contextlib.redirect_stdout(io.StringIO()):
        bandstructure.lowest_bands(h,nkpoints=nkp,nbands=4)
    outs.append(open("BANDS.OUT").read())
print("BANDS.OUT rows with nkpoints=8  :",outs[0].count("\n"))
print("BANDS.OUT rows with nkpoints=500:",outs[1].count("\n"))
print("byte-identical:",outs[0]==outs[1])
```

## `repro_magnetization.py`

Backs: **L5 siblings** — CLEARED: magnetism.compute_magnetization takes my = Im dm[2i,2i+1] from the transposed density matrix

```python
import numpy as np, pyqula, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from pyqula import geometry
np.random.seed(0)
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True)
h.add_zeeman([0.13,0.31,0.21])     # generic direction -- all three components nonzero
h.add_rashba(0.1)
nk = 40
# ---- oracle: explicit sum over occupied Bloch states ----
from pyqula import klist
hk = h.get_hk_gen()
sx = np.array([[0,1],[1,0]],dtype=complex)
sy = np.array([[0,-1j],[1j,0]],dtype=complex)
sz = np.array([[1,0],[0,-1]],dtype=complex)
ks = klist.kmesh(h.dimensionality,nk=nk)
evs=[]
for k in ks:
    m=np.array(hk(k)); e,w=np.linalg.eigh(m); evs.append((e,w))
allE=np.concatenate([e for e,_ in evs]); allE.sort()
ef=allE[len(allE)//2-1]   # half filling
acc=np.zeros(3)
for e,w in evs:
    for i in range(len(e)):
        if e[i]<=ef:
            v=w[:,i]
            acc[0]+= (np.conjugate(v)@(np.kron(np.eye(1),sx))@v).real/2
            acc[1]+= (np.conjugate(v)@(np.kron(np.eye(1),sy))@v).real/2
            acc[2]+= (np.conjugate(v)@(np.kron(np.eye(1),sz))@v).real/2
acc/=len(ks)
print("oracle  <sx>/2,<sy>/2,<sz>/2 per cell =",np.round(acc,6))
from pyqula.magnetism import compute_magnetization
mx,my,mz = compute_magnetization(h,nk=nk)
print("magnetism.compute_magnetization       =",np.round([mx[0],my[0],mz[0]],6))
print("ratio (code/oracle)                   =",np.round(np.array([mx[0],my[0],mz[0]])/acc,4))
# and spectrum.ev, the path fixed in 7087ee6
from pyqula import spectrum
sxo=h.get_operator("sx"); syo=h.get_operator("sy"); szo=h.get_operator("sz")
print("spectrum.ev(sx,sy,sz) (total, /2)     =",
      np.round(spectrum.ev(h,operator=[sxo.get_matrix(),syo.get_matrix(),szo.get_matrix()],nk=nk).real/2,6))
print("h.get_magnetization(mode='vev')       =",np.round(h.get_magnetization(nk=nk)[0],6))
```

## `repro_misc.py`

Backs: **L8 features/docs** — topology.operator_berry accepts ewindow= and ignores it - the sibling that bug_audit 2.2 fixed in bandstructure.py

Backs: **L8 features/docs** — Non-Hermitian h.get_dos accepts any mode string, including a typo, and silently returns ED

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry,topology
print("--- topology.operator_berry ewindow")
g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(); h.add_haldane(0.05)
a=topology.operator_berry(h,k=[0.2,0.3])
b=topology.operator_berry(h,k=[0.2,0.3],ewindow=lambda e: False)  # keep nothing
print("  no ewindow:",a,"  ewindow-that-keeps-nothing:",b," identical:",a==b)
print("  (bandstructure.get_bands honours the same kwarg:)")
n0=len(h.get_bands(nk=10)[1]); n1=len(h.get_bands(nk=10,ewindow=lambda e:abs(e)<0.5)[1])
print("  get_bands rows:",n0,"->",n1)
print("--- non-Hermitian get_dos ignores mode")
hn=geometry.chain().get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1])
hn.add_onsite(lambda r:0.2j)
es=np.linspace(-1,1,5)
for m in ["ED","KPM","adaptive","bogus"]:
    try:
        o=hn.get_dos(energies=es,mode=m,delta=0.1,nk=10)
        print("  NH mode=%-8s accepted -> %s"%(m,np.round(o[1],4)))
    except Exception as ex: print("  NH mode=%-8s %s: %s"%(m,type(ex).__name__,str(ex)[:80]))
print("--- hermitian control")
hh=geometry.chain().get_hamiltonian(has_spin=False)
for m in ["ED","bogus"]:
    try:
        o=hh.get_dos(energies=es,mode=m,delta=0.1,nk=10); print("  H mode=%-8s accepted"%m)
    except Exception as ex: print("  H mode=%-8s %s: %s"%(m,type(ex).__name__,str(ex)[:80]))
```

## `repro_mode_switch.py`

Backs: **L7 test coverage** — tests/densitymatrix/test_acceleration.py claims coverage of the explicit/vectorized density-matrix branches for the ds path, but that switch is ignored there and full_dm_python_d has no callers

```python
"""Do the module-level mode switches that tests/densitymatrix/test_acceleration.py
and tests/chi/test_acceleration.py toggle actually reach two different code paths?
Instrument the branch targets and count."""
import sys, numpy as np
sys.path.insert(0,"<repo root>/tests")
from pyqula import geometry
from pyqula.dmtk import fulldm
from pyqula.chitk import rpa
from testutils import temporary_attr, random_hermitian_hamiltonian

counts = {}
def wrap(mod,name):
    real = getattr(mod,name); counts[name]=0
    def f(*a,**k):
        counts[name]+=1; return real(*a,**k)
    setattr(mod,name,f); return real

# --- density matrix -------------------------------------------------
for nm in ("full_dm_explicit","full_dm_vectorized","full_dm_batch_vectorized",
           "full_dm_batch_d_vectorized","full_dm_d_batch_vectorized"):
    wrap(fulldm,nm)
import pyqula.densitymatrix as dmmod
# densitymatrix imported the names directly, re-bind those too
for nm in ("full_dm_batch_vectorized","full_dm_batch_d_vectorized","full_dm_d_batch_vectorized"):
    setattr(dmmod,nm,getattr(fulldm,nm))

np.random.seed(0)
h = random_hermitian_hamiltonian(geometry.honeycomb_lattice, supercell=2)
for use_ds in (True,False):
  for m1 in ("explicit","vectorized"):
    for m2 in ("accumulate","simultaneous"):
        for k in counts: counts[k]=0
        with temporary_attr(fulldm,"mode",m1):
            ds = [[i,0,0] for i in range(3)] if use_ds else None
            h.get_density_matrix(nk=3, ds=ds, dm_mode=m2)
        print(f"use_ds={use_ds!s:5} fulldm.mode={m1:11} dm_mode={m2:13} -> "
              + ", ".join(f"{k}={v}" for k,v in counts.items() if v))
```

## `repro_multildos.py`

```python
import numpy as np, pyqula, os
os.chdir("SCRATCH")
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False)
es = np.linspace(-3.,3.,61)
nk=60
h.get_multildos(energies=es,delta=0.05,nrep=1,nk=nk)
# DOS written by the same routine
dosf = np.genfromtxt("MULTILDOS/DOS.OUT").T
# sum over sites of the LDOS map, the same quantity
names = [l.strip() for l in open("MULTILDOS/MULTILDOS.TXT")]
tot = []
for n in names:
    d = np.genfromtxt("MULTILDOS/"+n)
    tot.append(np.atleast_2d(d)[:,2].sum())
tot=np.array(tot)
# compare at the same energies (DOS.OUT is on a 10x finer grid)
from scipy.interpolate import interp1d
f = interp1d(dosf[0],dosf[1])
ref = f(es)
print("sum_sites LDOS(e)   :",np.round(tot[25:31],5))
print("MULTILDOS/DOS.OUT(e):",np.round(ref[25:31],5))
print("ratio DOS.OUT / sum_sites LDOS =",np.round(ref[25:31]/tot[25:31],4))
print("nk =",nk,"   pi*nk =",np.pi*nk)
# and the independent oracle
x,y = h.get_dos(energies=es,mode="ED",delta=0.05,nk=400)
print("h.get_dos(ED)      :",np.round(np.array(y)[25:31],5))
print("ratio DOS.OUT/get_dos =",np.round(ref[25:31]/np.array(y)[25:31],3))
print("ratio sumLDOS/get_dos =",np.round(tot[25:31]/np.array(y)[25:31],3))
```

## `repro_multildos2.py`

```python
import numpy as np, pyqula, os
os.chdir("SCRATCH")
from pyqula import geometry, klist
g = geometry.chain(); h = g.get_hamiltonian(has_spin=False)
es = np.linspace(-3.,3.,61); nk=60; delta=0.05
h.get_multildos(energies=es,delta=delta,nrep=1,nk=nk)
names=[l.strip() for l in open("MULTILDOS/MULTILDOS.TXT")]
tot=np.array([np.atleast_2d(np.genfromtxt("MULTILDOS/"+n))[:,2].sum() for n in names])
nkactual = len(klist.kmesh(h.dimensionality,nk=nk))
# oracle: explicit Lorentzian sum over the same k-mesh, properly normalized
hk = h.get_hk_gen()
ev=np.concatenate([np.linalg.eigvalsh(np.array(hk(k))) for k in klist.kmesh(1,nk=nk)])
ref=np.array([np.sum(delta/((e-ev)**2+delta**2))/np.pi/nkactual for e in es])
print("nk points used =",nkactual)
print("site-summed LDOS / oracle  =",np.round(tot[25:31]/ref[25:31],4))
dosf=np.genfromtxt("MULTILDOS/DOS.OUT").T
from scipy.interpolate import interp1d
dd=interp1d(dosf[0],dosf[1])(es)
print("MULTILDOS/DOS.OUT / oracle =",np.round(dd[25:31]/ref[25:31],4))
print("expected if only 1/nk missing:",nkactual,"   if 1/nk and 1/pi missing:",nkactual*np.pi)
```

## `repro_multildos3.py`

```python
import numpy as np, pyqula, os
os.chdir("SCRATCH")
from pyqula import geometry
g = geometry.honeycomb_lattice(); h = g.get_hamiltonian(has_spin=False)
e=0.4; delta=0.05; nk=30
# the single-energy public LDOS
out = h.get_ldos(e=e,delta=delta,mode="arpack",nk=nk,nrep=1,write=False)
print("h.get_ldos (arpack) per site :",np.round(out[-1],6))
outg = h.get_ldos(e=e,delta=delta,mode="green",nk=nk,nrep=1,write=False)
print("h.get_ldos (green)  per site :",np.round(outg[-1],6))
h.get_multildos(energies=np.array([e]),delta=delta,nrep=1,nk=nk)
d=np.atleast_2d(np.genfromtxt("MULTILDOS/LDOS_"+str(e)+"_.OUT"))
print("h.get_multildos     per site :",np.round(d[:,2],6))
print("ratio multildos/arpack =",np.round(d[:,2]/np.array(out[-1]),3))
```

## `repro_nh2.py`

Backs: **L5 siblings** — Non-Hermitian get_bands_nd ignores write/output_file and raises NameError on num_bands

```python
import numpy as np, pyqula, os
os.chdir("SCRATCH")
for f in os.listdir("."): os.remove(f)
from pyqula import geometry
g = geometry.chain().get_supercell(6,store_primal=True)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True)
h.add_onsite(lambda r: 0.1j*np.cos(np.pi*2*r[0]/6.))
print("h.non_hermitian =",h.non_hermitian)
out = h.get_bands(nk=10,write=True,output_file="BANDS.OUT")
print("files after get_bands(write=True) :",sorted(os.listdir(".")))
h2 = geometry.chain().get_supercell(6).get_hamiltonian(has_spin=False)
h2.get_bands(nk=10,write=True,output_file="BANDS.OUT")
print("Hermitian reference writes        :",sorted(os.listdir(".")))
print("--- num_bands on a non-Hermitian Hamiltonian ---")
try:
    h.get_bands(nk=4,num_bands=3,write=False)
    print("   OK")
except Exception as e:
    print("   ->",type(e).__name__,":",str(e)[:150])
print("--- get_dos on a non-Hermitian Hamiltonian ---")
try:
    x,y = h.get_dos(energies=np.linspace(-3,3,20))
    print("   OK")
except Exception as e:
    print("   get_dos ->",type(e).__name__,":",str(e)[:200])
```

## `repro_nh_bands.py`

```python
import numpy as np, pyqula, os
os.chdir("SCRATCH")
for f in os.listdir("."): os.remove(f)
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False)
h.add_onsite(0.2j)             # non-Hermitian
print("h.non_hermitian =",h.non_hermitian)
out = h.get_bands(nk=10,write=True,output_file="BANDS.OUT")
print("files written by get_bands(write=True):",sorted(os.listdir(".")))
# Hermitian reference
h2 = g.get_hamiltonian(has_spin=False)
out2 = h2.get_bands(nk=10,write=True,output_file="BANDS.OUT")
print("Hermitian reference writes       :",sorted(os.listdir(".")))
print("--- num_bands on a non-Hermitian Hamiltonian ---")
g3 = geometry.honeycomb_lattice().get_supercell(4)
h3 = g3.get_hamiltonian(has_spin=False); h3.add_onsite(0.2j)
try:
    h3.get_bands(nk=4,num_bands=4,write=False)
    print("   OK")
except Exception as e:
    print("   ->",type(e).__name__,":",str(e)[:150])
```

## `repro_nh_bands2.py`

Backs: **L8 features/docs** — h.get_bands(num_bands=...) raises NameError on any non-Hermitian Hamiltonian - slg, arpack_tol and arpack_maxiter are undefined in nonhermitiantk/bandstructure.py

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry
g = geometry.chain().get_supercell(4)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1])
h.add_onsite(lambda r: 0.3j)
print("dim intra",h.intra.shape)
for nb in [2]:
    try:
        out = h.get_bands(nk=5,num_bands=nb)
        print("NH num_bands=%d OK"%nb)
    except Exception as ex:
        print("NH num_bands=%d FAIL: %s: %s"%(nb,type(ex).__name__,ex))
# hermitian control, same call
h2 = geometry.chain().get_supercell(4).get_hamiltonian(has_spin=False)
try:
    out = h2.get_bands(nk=5,num_bands=2)
    print("Hermitian num_bands=2 OK, nbands per k:",len(out[1])//5)
except Exception as ex:
    print("Hermitian num_bands=2 FAIL:",type(ex).__name__,ex)

h3 = geometry.chain().get_supercell(4).get_hamiltonian(has_spin=False)
h3.add_onsite(0.37)
print("Hermitian control num_bands=2:",np.array(h3.get_bands(nk=5,num_bands=2)).shape)
# and the raw-matrix operator branch (braket_wAw)
from pyqula import operators
A = np.identity(4,dtype=np.complex128)
try:
    o = h.get_operator(A); print("get_operator(matrix) ->",type(o))
except Exception as ex: print("get_operator FAIL",ex)
```

## `repro_ops.py`

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry, operatorlist
names=operatorlist.get_operator_names()
print(len(names),"operator names")
g=geometry.honeycomb_lattice()
h=g.get_hamiltonian(has_spin=True); h.setup_nambu_spinor()
bad=[]
for n in names:
    try:
        o=h.get_operator(n)
        _=h.get_bands(nk=3,operator=n)
    except Exception as e: bad.append((n,type(e).__name__,str(e)[:60]))
for b in bad: print(" ",b)
print("failing:",len(bad),"of",len(names))
```

## `repro_pairing_modes.py`

Backs: **L8 features/docs** — h.add_pairing(mode="deltaud") raises NameError - one of the 22 pairing modes listed in pairing_generator's own elif chain is dead

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry
modes=["swave","deltaud","extended_swave","triplet","pwave","nodal_fwave",
 "chiral_pwave","chiral_fwave","chiral_dwave","chiral_gwave","antihaldane",
 "haldane","swavez","px","dpid","swaveA","swaveB","swavesublattice","dxy",
 "snn","C3nn","SnnAB"]
g=geometry.honeycomb_lattice()
bad=[]
for m in modes:
    h=g.get_hamiltonian(); h.setup_nambu_spinor()
    try:
        h.add_pairing(delta=0.1,mode=m); print("  %-20s OK"%m)
    except Exception as e:
        print("  %-20s %s: %s"%(m,type(e).__name__,str(e)[:70])); bad.append((m,type(e).__name__))
print("BROKEN:",bad)
```

## `repro_piflux.py`

Backs: **L5 siblings** — specialhamiltonian.triangular_pi_flux() cannot return a Hamiltonian for any input

```python
import numpy as np, pyqula
from pyqula import specialhamiltonian
print("about to call specialhamiltonian.triangular_pi_flux() ...")
h = specialhamiltonian.triangular_pi_flux()
print("RETURNED:",h)          # never reached
print("bands:",h.get_bands(nk=4,write=False)[1][:3])
```

## `repro_print_h.py`

Backs: **L5 siblings** — h.print_hamiltonian() raises AttributeError on every multicell (2d/3d) Hamiltonian

```python
import pyqula
from pyqula import geometry
for name,g in [("chain",geometry.chain()),("honeycomb",geometry.honeycomb_lattice())]:
    h=g.get_hamiltonian()
    try:
        h.print_hamiltonian(); print(name,"OK")
    except Exception as e: print(name,"->",type(e).__name__,":",str(e)[:120])
```

## `repro_realspace_chern.py`

Backs: **L7 test coverage** — tests/topology/test_real_space_chern_island.py is vacuous by a matrix identity: sum(marker) is Tr of a commutator, identically zero for every Hamiltonian

```python
"""tests/topology/test_real_space_chern_island.py asserts
    np.isclose(np.sum(c), 1.0658141036401503e-14, atol=1e-6)
The marker is C_i = 2*pi*Im diag([PXP, PYP])_ii / scale (topologytk/realspace.py:27-31).
The trace of ANY commutator is exactly zero, so sum(c) is zero by a matrix identity,
for every Hamiltonian, every Haldane mass, topological or not.  The test cannot fail.
The physics lives in the BULK value of the marker, which the test never looks at.
"""
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import islands, topology, geometry

g = islands.get_geometry(name="honeycomb", n=6, nedges=4, rot=0.0, clean=False)
for lab,build in [
        ("Haldane 0.1 (the test's own system, C=+1)", lambda h: h.add_haldane(0.1)),
        ("Haldane 0.0 -- NOT topological at all",     lambda h: None),
        ("Haldane -0.1 -- OPPOSITE Chern number",     lambda h: h.add_haldane(-0.1)),
        ("Haldane 0.5 -- different magnitude",        lambda h: h.add_haldane(0.5)),
        ("sublattice mass 0.4 -- trivial insulator",  lambda h: h.add_sublattice_imbalance(0.4)),
        ]:
    h = g.get_hamiltonian(has_spin=False); build(h)
    r,c = topology.real_space_chern(h)
    # bulk region: the marker averaged over the inner third, the quantity that
    # should equal the Chern number
    rr = np.array(r); d2 = np.sum(rr*rr,axis=1); cut = np.max(d2)/3.
    bulk = np.mean(np.array(c)[d2<cut])
    print(f"{lab:45s} sum(c)={np.sum(c):+.3e}   "
          f"test passes: {np.isclose(np.sum(c),1.0658141036401503e-14,atol=1e-6)}"
          f"   bulk mean marker={bulk:+.4f}")

print()
print("independent oracle available in-repo: the periodic Haldane Chern number")
for hal in (0.1,-0.1,0.0):
    hp = geometry.honeycomb_lattice().get_hamiltonian(has_spin=False)
    if hal!=0.: hp.add_haldane(hal)
    print(f"  add_haldane({hal:+.1f}) -> h.get_chern(nk=10) = {hp.get_chern(nk=10):+.4f}")
```

## `repro_remove_sites.py`

Backs: **L5 siblings** — h.remove_sites corrupts a spinful Hamiltonian before raising its NotImplementedError

```python
import numpy as np, pyqula
assert "/src/pyqula" in pyqula.__file__
from pyqula import geometry
g = geometry.honeycomb_lattice()
g = g.get_supercell(2)
h = g.get_hamiltonian(has_spin=True)   # spinful, the DEFAULT
print("before: nsites =",len(h.geometry.r)," intra.shape =",h.intra.shape)
store = np.ones(len(h.geometry.r),dtype=int); store[0]=0
try:
    h.remove_sites(store)
except NotImplementedError as e:
    print("raised NotImplementedError:",e)
print("after : nsites =",len(h.geometry.r)," intra.shape =",h.intra.shape)
print("consistent?", h.intra.shape[0]==2*len(h.geometry.r))
try:
    (k,e)=h.get_bands(nk=3,write=False)
    print("get_bands OK")
except Exception as ex:
    print("get_bands ->",type(ex).__name__,":",str(ex)[:200])

print("--- downstream silent garbage ---")
d = h.get_density(nk=2)
print("len(density) =",len(d)," len(geometry.r) =",len(h.geometry.r))
```

## `repro_rpa_mode.py`

Backs: **L7 test coverage** — CLEARED: tests/chi/test_acceleration.py comparing rpa.mode_rpa 'sequential' against 'vectorized'

```python
"""tests/chi/test_acceleration.py toggles chitk.rpa.mode_rpa between
'sequential' and 'vectorized' and compares h.get_qdos_iets output.
Does get_qdos_iets ever reach the branch?"""
import sys, numpy as np
sys.path.insert(0,"<repo root>/tests")
from pyqula import geometry
from pyqula.chitk import rpa
from testutils import temporary_attr, random_hermitian_hamiltonian

n = {"chi_ops_RPA":0,"_chi_ops_matrix_vectorized":0}
real_rpa = rpa.chi_ops_RPA
def spy(*a,**k):
    n["chi_ops_RPA"]+=1; return real_rpa(*a,**k)
rpa.chi_ops_RPA = spy
real_vec = rpa._chi_ops_matrix_vectorized
def spy2(*a,**k):
    n["_chi_ops_matrix_vectorized"]+=1; return real_vec(*a,**k)
rpa._chi_ops_matrix_vectorized = spy2

np.random.seed(0)
h = random_hermitian_hamiltonian(geometry.bichain, supercell=2)
E = np.linspace(0.,5.0,40)
outs = {}
for mode in ("sequential","vectorized"):
    for k in n: n[k]=0
    with temporary_attr(rpa,"mode_rpa",mode):
        _,_,chis = h.get_qdos_iets(nk=4,nq=6,energies=E)
    outs[mode]=np.array(chis)
    print(f"mode_rpa={mode:12} -> chi_ops_RPA calls={n['chi_ops_RPA']}, "
          f"_chi_ops_matrix_vectorized calls={n['_chi_ops_matrix_vectorized']}")
d = np.max(np.abs(outs["sequential"]-outs["vectorized"]))
print("max |sequential - vectorized| =", d, " (exactly zero =>", d==0.0, ")")
```

## `repro_selected_bands2d_negative.py`

Backs: **L6 optimization** — selected_bands2d's negative-band branch reports the conduction state's operator expectation next to the valence band's energy

```python
"""LENS6: spectrum.selected_bands2d's negative-index branch (the DEFAULT
nindex=[-1,1]) writes the VALENCE energy but the CONDUCTION eigenvector's
operator expectation value: `op.braket(wfpos[abs(i)-1])` where the energy
came from eneg. It also writes a '\n' before the operator columns, so the
row is split in two. Oracle: an explicit <psi|sz|psi> on the occupied
state at the same k-point."""
import os, tempfile, numpy as np
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, spectrum

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(); h.add_zeeman([0.,0.,0.25]); h.add_rashba(0.2)
op = h.get_operator("sz")
spectrum.selected_bands2d(h, nindex=[-1,1], operator=[op], nk=3,
        reciprocal=False)
raw = open("BANDS2D__-1.OUT").read()
print("first 4 lines of BANDS2D__-1.OUT (note the split rows):")
for l in raw.split("\n")[:4]: print("   ", repr(l))

# explicit reference at one k-point
kxs = np.linspace(-1,1,3)
k = np.array([kxs[0],kxs[0],0.])
hk = h.get_hk_gen()(k)
es,ws = np.linalg.eigh(hk); ws = ws.T
eneg = sorted([e for e in es if e<0])[::-1]  # as the code sorts them
epos = sorted([e for e in es if e>0])
wneg = [w for (e,w) in sorted(zip(es,ws.tolist()),key=lambda p:p[0]) if e<0]
wpos = [w for (e,w) in sorted(zip(es,ws.tolist()),key=lambda p:p[0]) if e>0]
M = op.get_matrix()
sz_valence  = np.conj(wneg[-1])@(M@np.array(wneg[-1]))
sz_conduct  = np.conj(wpos[0])@(M@np.array(wpos[0]))
print("\nat k =",k[:2])
print("  <sz> of the highest OCCUPIED state (what -1 should report):",
      np.round(sz_valence.real,10))
print("  <sz> of the lowest EMPTY state (what the code reports):    ",
      np.round(sz_conduct.real,10))
print("  value written in the file:", raw.split("\n")[1].strip())
```

## `repro_sparse_batch_trap.py`

Backs: **L6 optimization** — current_bands stacks H(k) with a plain np.array, bypassing hk_matrix_batch — TypeError on any sparse Hamiltonian

Backs: **L6 optimization** — spectrum.ev2d's sparse branch references an undefined name `nindex` — NameError on any sparse Hamiltonian

```python
"""LENS6 / HARD RULE 2: find every batched H(k) stack that does NOT go
through htk.eigenvectors.hk_matrix_batch, and show the latent crash."""
import os, tempfile, numpy as np
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, bandstructure, spectrum

g = geometry.chain()
h = g.get_hamiltonian()
h.add_zeeman([0.,0.,0.3])
hs = h.copy(); hs.turn_sparse()
print("is_sparse:", h.is_sparse, hs.is_sparse)

# --- 1. bandstructure.current_bands (bandstructure.py:52) ---
print("\n[current_bands] dense:")
try:
    bandstructure.current_bands(h, klist=np.linspace(0,1.,5))
    print("  OK")
except Exception as e:
    print("  %s: %s"%(type(e).__name__, e))
print("[current_bands] sparse:")
try:
    bandstructure.current_bands(hs, klist=np.linspace(0,1.,5))
    print("  OK")
except Exception as e:
    print("  %s: %s"%(type(e).__name__, e))

# --- 2. spectrum.ev2d / selected_bands2d sparse branch ---
g2 = geometry.honeycomb_lattice()
h2 = g2.get_hamiltonian(); h2.add_zeeman([0.,0.,0.3])
h2s = h2.copy(); h2s.turn_sparse()
print("\n[ev2d] dense:")
try:
    spectrum.ev2d(h2, nk=3); print("  OK")
except Exception as e: print("  %s: %s"%(type(e).__name__, e))
print("[ev2d] sparse:")
try:
    spectrum.ev2d(h2s, nk=3); print("  OK")
except Exception as e: print("  %s: %s"%(type(e).__name__, e))
print("\n[selected_bands2d] sparse:")
try:
    spectrum.selected_bands2d(h2s, nindex=[1,-1], nk=3); print("  OK")
except Exception as e: print("  %s: %s"%(type(e).__name__, e))
```

## `repro_spin_splitting.py`

Backs: **L7 test coverage** — h.get_average_spin_splitting has zero tests and two defects: it is not intensive (sums over bands), and it lacks the collinearity guard its tested sibling has

```python
"""h.get_average_spin_splitting (fermisurfacetk/spinsplitting.average_spin_splitting)
has ZERO tests and zero examples.  Two defects, both caught by oracles its own
tested sibling spin_splitting_vs_energy already uses.

(1) It SUMS over bands instead of averaging, so the "average spin splitting"
    is not intensive: describing the SAME physical system in a supercell
    multiplies the answer by the number of repetitions.
(2) It has no check_collinear guard, although remove_spin silently drops the
    spin off-diagonal block.  spin_splitting_vs_energy refuses such a
    Hamiltonian outright; average_spin_splitting returns a plausible number.
"""
import numpy as np
from pyqula import geometry

print("--- (1) not intensive: same physics, bigger cell -------------------")
for ns in (1,2,3,4):
    g = geometry.honeycomb_lattice()
    if ns>1: g = g.get_supercell(ns)
    h = g.get_hamiltonian()
    h.add_zeeman([0.,0.,0.3])          # uniform field: splitting is 0.6 everywhere
    v = h.get_average_spin_splitting(nk=6)
    norb = h.intra.shape[0]//2         # bands per spin channel
    print(f"  supercell={ns}  bands/spin={norb:3d}  average_spin_splitting={v:.6f}"
          f"   (= {v/norb:.6f} per band; the true splitting is 0.600000)")

print()
print("--- exact per-band splitting from the spectrum (the oracle) --------")
h = geometry.honeycomb_lattice().get_hamiltonian(); h.add_zeeman([0.,0.,0.3])
hup=h.copy(); hup.remove_spin(channel="up"); hdn=h.copy(); hdn.remove_spin(channel="dn")
k=[0.31,0.17,0.]
eu=np.sort(np.linalg.eigvalsh(np.array(hup.get_hk_gen()(k))))
ed=np.sort(np.linalg.eigvalsh(np.array(hdn.get_hk_gen()(k))))
print("  E_up - E_dn per band at a generic k =",eu-ed)

print()
print("--- (2) no collinearity guard, unlike the tested sibling -----------")
h = geometry.honeycomb_lattice().get_hamiltonian()
h.add_rashba(0.3)                      # spin is NOT a good quantum number
print("  average_spin_splitting(Rashba)   ->",h.get_average_spin_splitting(nk=6))
try:
    h.get_spin_splitting_vs_energy(nk=6)
    print("  spin_splitting_vs_energy(Rashba) -> returned a number")
except Exception as e:
    print("  spin_splitting_vs_energy(Rashba) ->",type(e).__name__+":",str(e)[:90])
try:
    print("  spin_splitting_density(Rashba)   ->",
          np.max(h.get_spin_splitting_density(nk=6)[1]))
except Exception as e:
    print("  spin_splitting_density(Rashba)   ->",type(e).__name__+":",str(e)[:90])
```

## `repro_sum_is_zero.py`

Backs: **L7 test coverage** — ~28 assertions in ~20 test files pin `sum(bands) == 0`, which is Tr H(k) — guaranteed by tracelessness, blind to the model each test names

```python
"""~25 assertions across ~15 test files pin `np.sum(<quantity>) == <1e-13ish>`
with atol=1e-6, i.e. they assert "this sum is zero".  For band energies the sum
over a k-path is sum_k Tr H(k); on a bipartite lattice with no onsite term every
hopping matrix is off-diagonal in the site index, so the trace is zero for ANY
parameter values.  The recorded constant therefore carries one bit (traceless),
not the band structure, and the test is blind to the model it names.
"""
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import geometry

REF = {"plain":1.2079226507921703e-13,
       "km":1.021405182655144e-13,
       "hald":-9.769962616701378e-15}

def s(h):
    k,e = h.get_bands()
    return np.sum(e)

print("=== tests/ribbon/test_armchair_ribbon_bands.py ===")
cases = [
  ("the test's own system: width 10, plain",      lambda: geometry.honeycomb_armchair_ribbon(10).get_hamiltonian()),
  ("width 4 instead of 10 (different system!)",   lambda: geometry.honeycomb_armchair_ribbon(4).get_hamiltonian()),
  ("width 10, Kane-Mele 0.1 (test 2's system)",   lambda: _km(10,0.1)),
  ("width 10, Kane-Mele 9.9 (100x too strong)",   lambda: _km(10,9.9)),
  ("width 10, Haldane 5.0",                       lambda: _hal(10,5.0)),
  ("width 10, Rashba 2.0 + Kane-Mele 0.1",        lambda: _rash(10)),
]
def _km(n,v):
    h = geometry.honeycomb_armchair_ribbon(n).get_hamiltonian(); h.add_kane_mele(v); return h
def _hal(n,v):
    h = geometry.honeycomb_armchair_ribbon(n).get_hamiltonian(has_spin=True); h.add_haldane(v); return h
def _rash(n):
    h = geometry.honeycomb_armchair_ribbon(n).get_hamiltonian()
    h.add_kane_mele(0.1); h.add_rashba(2.0); return h
for lab,f in cases:
    v = s(f())
    ok = [k for k,r in REF.items() if np.isclose(v,r,atol=1e-6)]
    print(f"  {lab:42s} sum(e)={v:+.3e}   passes all 3 recorded refs: {len(ok)==3}")

print()
print("=== the one thing that DOES break it: a nonzero trace ===")
h = geometry.honeycomb_armchair_ribbon(10).get_hamiltonian(); h.add_onsite(0.3)
print(f"  add_onsite(0.3) -> sum(e)={s(h):+.4f}  (so the assertion only sees Tr H)")
```

## `repro_surface_dos_op.py`

Backs: **L8 features/docs** — dos.surface_dos accepts operator= and silently ignores it, while dos.get_dos in the same module honours it

```python
import sys,numpy as np
sys.path.insert(0,"<repo root>/src")
from pyqula import geometry,heterostructures,dos
g = geometry.chain()
h = g.get_hamiltonian(has_spin=True)
h.add_exchange([0.,0.,0.6])   # spin-polarized: sz-projected DOS must differ from total
es = np.linspace(-1.,1.,7)
e0,d0 = dos.surface_dos(h,energies=es,delta=0.05)
e1,d1 = dos.surface_dos(h,energies=es,delta=0.05,operator="sz")
print("surface_dos operator=None:",np.round(d0,6))
print("surface_dos operator=sz  :",np.round(d1,6))
print("byte identical:",np.array_equal(d0,d1))
# the sibling that DOES honour it
from pyqula import dos as D
b0 = D.get_dos(h,energies=es,delta=0.05)
b1 = D.get_dos(h,energies=es,delta=0.05,operator="sz")
print("bulk get_dos None:",np.round(b0[1],5))
print("bulk get_dos sz  :",np.round(b1[1],5))
print("bulk identical:",np.array_equal(b0[1],b1[1]))
```

## `repro_swallow.py`

Backs: **L5 siblings** — h.get_multildos(operator=...) is silently swallowed because the parameter is spelled op

Backs: **L5 siblings** — vev.get_dm_vev drops every keyword argument it accepts

```python
import numpy as np, pyqula, os, io, contextlib
os.chdir("SCRATCH")
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True); h.add_zeeman([0.,0.,0.6])
es = np.array([0.3])
def run(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        h.get_multildos(energies=es,delta=0.05,nrep=1,nk=6,**kw)
    return np.genfromtxt("MULTILDOS/LDOS_0.3_.OUT")
a = run()
b = run(operator="sz")
c = run(op=h.get_operator("sz"))
print("get_multildos(operator='sz') identical to no operator :",np.array_equal(a,b))
print("get_multildos(op=<sz>)       identical to no operator :",np.array_equal(a,c))
print("   (every other LDOS/DOS entry point spells it 'operator')")
print("   h.get_ldos(operator='sz') differs from plain       :",
      not np.array_equal(h.get_ldos(e=0.3,delta=0.05,nk=6,nrep=1,write=False,operator="sz")[-1],
                         h.get_ldos(e=0.3,delta=0.05,nk=6,nrep=1,write=False)[-1]))

print()
print("### get_dm_vev swallows T ###")
from pyqula import islands
g0 = islands.get_geometry(n=1,nedges=3,rot=0.)
h0 = g0.get_hamiltonian(has_spin=True); h0.add_zeeman([0.,0.,0.4])
v0 = h0.get_dm_vev(h0.get_operator("sz").get_matrix())
v1 = h0.get_dm_vev(h0.get_operator("sz").get_matrix(),T=2.0)
print("   get_dm_vev('sz')        =",np.round(v0,8))
print("   get_dm_vev('sz',T=2.0)  =",np.round(v1,8),"  identical:",v0==v1)
print("   h.get_density_matrix(T=2.0) differs from T=0:",
      not np.array_equal(h0.get_density_matrix(),h0.get_density_matrix(T=2.0)))
```

## `repro_vacuous_eigvec.py`

Backs: **L7 test coverage** — The 'phase-invariant density-matrix sum' used as the eigenvector oracle in two parallel tests is identically nk*Identity, so it passes on random unitaries

```python
"""The 'phase-invariant density-matrix sum' used as the eigenvector oracle in
tests/parallel/test_get_eigenvectors_dense.py and
tests/parallel/test_six_more_thread_independence.py is conj(vs).T @ vs where
vs has one eigenvector per ROW, over all k and all bands.  For a complete
orthonormal set at each k that is identically nk*Identity, independent of the
eigenvectors.  So the assertion cannot fail -- it does not compare the two
code paths at all."""
import numpy as np
from pyqula import geometry
from pyqula.htk.eigenvectors import get_eigenvectors

g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(); h.add_zeeman([0.,0.,0.3])
nk = 5
es, vs = get_eigenvectors(h, nk=nk)
dm = np.conj(vs).T @ vs
nkpts = vs.shape[0]//h.intra.shape[0]
print("vs.shape          =", vs.shape, " -> nkpoints =", nkpts)
print("max|dm - nk*I|    =", np.max(np.abs(dm - nkpts*np.eye(dm.shape[0]))))

# Now feed the SAME check completely GARBAGE eigenvectors: any random unitary
# per k-point passes it identically.
rng = np.random.default_rng(0)
n = h.intra.shape[0]
fake = []
for _ in range(nkpts):
    a = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    q,_ = np.linalg.qr(a)              # a random unitary, nothing to do with h
    for row in q.T: fake.append(row)
fake = np.array(fake)
dm_fake = np.conj(fake).T @ fake
print("max|dm_true - dm_fake| =", np.max(np.abs(dm - dm_fake)))
print("np.allclose(dm_true, dm_fake, atol=1e-8) ->",
      np.allclose(dm, dm_fake, atol=1e-8),
      "  <-- the test's exact assertion, passing on random unitaries")
```

## `s10_misc.py`

```python
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology

print("== berry_phase sign vs the standard King-Smith/Vanderbilt discrete formula ==")
# 1D two-site chain with alternating hoppings (SSH-like), via a 2-site supercell
g=geometry.chain(); g=g.supercell(2)
h=g.get_hamiltonian(has_spin=False)
def fun(r1,r2):
    d=r1-r2
    if 0.9<np.linalg.norm(d)<1.1:
        return 1.0 if (round(min(r1[0],r2[0]))%2==0) else 0.4
    return 0.0
h=g.get_hamiltonian(has_spin=False,fun=fun,is_sparse=False)
h.shift_fermi(0.0)
nk=400
hk=h.get_hk_gen()
us=[]
for i in range(nk):
    m=hk([i/nk,0.,0.]); m=np.array(m.todense()) if hasattr(m,'todense') else np.array(m)
    es,ws=np.linalg.eigh(m)
    us.append(ws[:,es<0.])
prod=1.+0j
for i in range(nk):
    a=us[i]; b=us[(i+1)%nk]
    prod*=np.linalg.det(np.conjugate(a.T)@b)
gamma=-np.angle(prod)   # KSV: gamma = -Im log prod <u_j|u_{j+1}>
print("  my KSV gamma/pi          =",gamma/np.pi)
print("  topology.berry_phase/pi  =",topology.berry_phase(h,nk=nk)/np.pi)

print("\n== Z2 on Kane-Mele ==")
g=geometry.honeycomb_lattice(); hkm=g.get_hamiltonian(has_spin=True); hkm.add_kane_mele(0.1)
print("  z2 soc only   :",topology.z2_invariant(hkm,nk=20,nt=20))
h2=hkm.copy(); h2.add_sublattice_imbalance(1.5)
print("  z2 large mass :",topology.z2_invariant(h2,nk=20,nt=20))

print("\n== nodes.dirac_points on graphene ==")
from pyqula import nodes
g=geometry.honeycomb_lattice(); hg=g.get_hamiltonian(has_spin=False)
try:
    np.random.seed(1); nodes.dirac_points(hg)
except Exception as e:
    print("  EXC",type(e).__name__,str(e)[:200])

print("\n== mass.effective_mass on a 1d chain (analytic m for e=-2t cos(2pi k)) ==")
from pyqula import mass
gc=geometry.chain(); hc=gc.get_hamiltonian(has_spin=False)
for k in ([0.,0.,0.],[0.25,0.,0.]):
    try:
        print("  k=",k,"effective_mass ->",mass.effective_mass(hc,np.array(k)))
    except Exception as e:
        print("  k=",k,"EXC",type(e).__name__,str(e)[:150])
```

## `s10b_berryphase.py`

```python
"""berry_phase sign: compare to the standard King-Smith & Vanderbilt discrete
Berry phase gamma = -Im log prod_j <u_j|u_{j+1}>, on the same closed k-path."""
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology
g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=False)
h.add_haldane(0.1); h.add_sublattice_imbalance(0.2)
hk=h.get_hk_gen()
def dense(m): return np.array(m.todense()) if hasattr(m,'todense') else np.array(m)
for R,ctr in [(0.10,[1/3.,1/3.]),(0.25,[1/3.,1/3.]),(0.05,[0.2,0.7])]:
    n=200
    path=[[ctr[0]+R*np.cos(2*np.pi*i/n), ctr[1]+R*np.sin(2*np.pi*i/n), 0.] for i in range(n)]
    us=[]
    for k in path:
        es,ws=np.linalg.eigh(dense(hk(k))); us.append(ws[:,es<0.])
    prod=1.+0j
    for i in range(n):
        prod*=np.linalg.det(np.conjugate(us[i].T)@us[(i+1)%n])
    ksv=-np.angle(prod)
    pyq=topology.berry_phase(h,kpath=np.array(path),write=False)
    print(f"R={R} ctr={ctr}: KSV gamma/pi = {ksv/np.pi: .6f}   topology.berry_phase/pi = {pyq/np.pi: .6f}")
```

## `s11_operator_kw.py`

Backs: **L4 topology** — operator= is resolved through topology.get_operator in only one of five Berry/Chern entry points; a string or matrix operator raises TypeError in the other four

Backs: **L4 topology** — topology.hall_conductivity's real implementation is dead code, shadowed 540 lines later by `hall_conductivity = chern`

```python
"""FINDING: operator= is resolved through topology.get_operator in only ONE of
the five Berry/Chern entry points (get_berry_curvature_path / write_berry).
mesh_chern/chern/h.get_chern, get_berry_curvature_master/h.get_berry_curvature,
chern_qtci and chern_density all pass the raw argument to
topologytk.green.berry_green, whose `omega = operator(omega,k=k)` needs a
CALLABLE -- so a string name or a plain matrix (what operators.get_sz returns)
raises TypeError instead of selecting the operator.  topology.get_operator
exists precisely to accept those forms and was fixed for string dispatch in
5f3da27, but only one caller uses it."""
import numpy as np, os, tempfile, io, contextlib
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology, operators
g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=True)
h.add_kane_mele(0.1); h.add_sublattice_imbalance(0.2)
szm=operators.get_sz(h); szo=h.get_operator('sz')
cases=[('h.get_chern',lambda o: h.get_chern(operator=o,nk=2,delta=0.2)),
       ('h.get_berry_curvature',lambda o: h.get_berry_curvature(operator=o,nk=2,delta=0.2,write=False)),
       ('topology.chern_density',lambda o: topology.chern_density(h,nk=1,operator=o,es=np.linspace(-1,1,2))),
       ('topology.chern_qtci',lambda o: topology.chern_qtci(h,operator=o,nk=4,delta=0.2)),
       ('topology.write_berry',lambda o: topology.get_berry_curvature_path(h,operator=o,nk=2))]
for lab,fn in cases:
  for tag,o in [("str 'sz'",'sz'),('matrix',szm),('Operator',szo)]:
    buf=io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): fn(o)
        print('%-26s %-9s OK'%(lab,tag))
    except Exception as e:
        print('%-26s %-9s %s: %s'%(lab,tag,type(e).__name__,str(e)[:60]))
```

## `s12_spintexture.py`

Backs: **L4 topology** — spintexture.kfun_map builds a ky grid and never uses it (the inner loop iterates kxs)

```python
"""FINDING: spintexture.kfun_map builds kys but never uses them -- the inner
loop is `for y in kxs`. With k0[1] != k0[0] the whole map is computed on the
wrong y grid, and the ky values returned/saved are wrong too."""
import numpy as np, io, contextlib
from pyqula import geometry, spintexture
g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=True); h.add_rashba(0.3)
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    kx,ky,out=spintexture.kfun_map(h,nk=4,operator=lambda m: 0.0,k0=[0.0,0.5])
print('k0=[0.0,0.5]: unique ky returned =',np.unique(np.round(np.array(ky),4)))
print('              expected           =',np.linspace(-1,1,4)+0.5)
```

## `s13_green_operator.py`

Backs: **L4 topology** — CLEARED: The Green's-function operator-projected Berry path (berry_green with an operator) -- the surface the four recorded-constant tests actually pin

```python
"""Value-check the Green's-function operator-projected Berry curvature
(topologytk.green.berry_green), which is the path write_berry / h.get_chern
actually use with an operator -- against the now-validated operator_berry."""
import numpy as np, os, tempfile, io, contextlib
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology, operators

g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=True)
h.add_kane_mele(0.1); h.add_sublattice_imbalance(0.2)
szm=operators.get_sz(h); szo=h.get_operator("sz")
f=h.get_gk_gen(delta=0.01)
def gI(e=0.,k=[0.,0.,0.]): return f(e=e,k=k,inv=True)
print("A. Kane-Mele, sz-projected curvature, pointwise")
print("   k                 berry_green(sz)   operator_berry(sz)   ratio")
for k in ([0.17,0.41,0.],[0.28,0.30,0.],[0.05,0.90,0.]):
    bg=topology.berry_green(f,k=k,operator=szo,gI=gI)
    ob=topology.operator_berry(h,k=k[:2],operator=szm)
    print(f"   {k[:2]}  {bg: 14.5f}  {ob: 16.5f}   {bg/ob: .5f}")
print("   (note berry_green already divides by 2*pi in green.py:fint)")
print("   ratio*2pi:", [round(topology.berry_green(f,k=k,operator=szo,gI=gI)/topology.operator_berry(h,k=k[:2],operator=szm)*2*np.pi,5) for k in ([0.17,0.41,0.],[0.28,0.30,0.])])

print("\nB. valley Chern of gapped graphene (examples/2d/valley_chern), analytic +-1")
g=geometry.honeycomb_lattice(); h2=g.get_hamiltonian(has_spin=False)
h2.add_sublattice_imbalance(0.1)
op=h2.get_operator("valley")
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    c=topology.chern(h2,mode="Green",delta=0.0001,nk=8,operator=op)
print("   valley Chern (nk=8, delta=1e-4) =",c)
with contextlib.redirect_stdout(buf):
    c0=topology.chern(h2,mode="Green",delta=0.0001,nk=8)
print("   total  Chern (nk=8, Green)      =",c0)
```

## `s14_valley_chern_conv.py`

Backs: **L4 topology** — CLEARED: The Green's-function operator-projected Berry path (berry_green with an operator) -- the surface the four recorded-constant tests actually pin

```python
import numpy as np, os, tempfile, io, contextlib, sys
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology, parallel
g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=False)
h.add_sublattice_imbalance(0.1)
for proj in [False,True]:
    op=h.get_operator("valley",projector=proj)
    for nk in [8,14,20,30]:
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            c=topology.chern(h,mode="Green",delta=0.0001,nk=nk,operator=op)
        print("projector=%s nk=%2d -> valley Chern = %.5f"%(proj,nk,c)); sys.stdout.flush()
```

## `s1_pointwise_kubo.py`

Backs: **L4 topology** — CLEARED: operator_berry's b*pi*pi*8 vs multicell.derivative's missing 2*pi -- do they still cancel exactly?

```python
"""Oracle 1: pointwise Kubo Berry curvature, built from scratch with a
finite-difference dH/dk of h.get_hk_gen(), vs topology.operator_berry."""
import numpy as np
from pyqula import geometry, topology

def haldane(t2=0.1,mass=0.2,spin=False):
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=spin)
    h.add_haldane(t2)
    if mass!=0.: h.add_sublattice_imbalance(mass)
    return h

def kubo_from_scratch(h,k,dk=1e-5,operator=None):
    hk = h.get_hk_gen()
    k = np.array(list(k)+[0.]*(3-len(k)))
    dx = np.array([dk,0.,0.]); dy = np.array([0.,dk,0.])
    vx = (np.array(hk(k+dx).todense() if hasattr(hk(k+dx),'todense') else hk(k+dx))
          - np.array(hk(k-dx).todense() if hasattr(hk(k-dx),'todense') else hk(k-dx)))/(2*dk)
    vy = (np.array(hk(k+dy).todense() if hasattr(hk(k+dy),'todense') else hk(k+dy))
          - np.array(hk(k-dy).todense() if hasattr(hk(k-dy),'todense') else hk(k-dy)))/(2*dk)
    m = hk(k)
    m = np.array(m.todense()) if hasattr(m,'todense') else np.array(m)
    es,ws = np.linalg.eigh(m)
    ws = ws.T  # rows are eigenvectors
    if operator is not None:
        O = np.array(operator.todense()) if hasattr(operator,'todense') else np.array(operator)
        vxo = (O@vx + vx@O)/2.
    else:
        vxo = vx
    n = len(es)
    V1 = np.conjugate(ws)@vxo@ws.T   # <n|vxo|m>
    V2 = np.conjugate(ws)@vy@ws.T    # <n|vy|m>
    om = np.zeros(n)
    for i in range(n):
        s = 0.
        for j in range(n):
            if i==j: continue
            s += np.imag(V1[i,j]*V2[j,i])/(es[i]-es[j])**2
        om[i] = -2.*s   # RMP Xiao-Chang-Niu convention
    return es,om

for (t2,mass) in [(0.1,0.2),(0.1,0.0),(-0.15,0.3)]:
    h = haldane(t2,mass)
    for k in ([0.17,0.41],[0.3333,0.3333],[0.05,0.9]):
        es,om = kubo_from_scratch(h,k)
        mine = np.sum(om[es<=0.])
        theirs = topology.operator_berry(h,k=k)
        print(f"t2={t2} mass={mass} k={k}  scratch={mine: .8f}  pyqula={theirs: .8f}  ratio={theirs/mine: .6f}")
```

## `s2_wilson_vs_kubo.py`

```python
"""Oracle 2: my own Fukui-Hatsugai-Suzuki plaquette Chern + pyqula's
berry_curvature vs the RMP Kubo curvature."""
import numpy as np
from pyqula import geometry, topology
import sys
sys.path.insert(0,'SCRATCH')
from s1_pointwise_kubo import kubo_from_scratch, haldane

def dense(m):
    return np.array(m.todense()) if hasattr(m,'todense') else np.array(m)

def fhs_chern(h,nk=24):
    """From-scratch Fukui-Hatsugai-Suzuki. C = (1/2pi) sum_plaq Im log U1U2U1*U2*
    with the STANDARD convention: F = Im log [U1(k)U2(k+1)U1(k+2)^-1 U2(k)^-1]."""
    hk = h.get_hk_gen()
    occ = {}
    for i in range(nk):
        for j in range(nk):
            m = dense(hk([i/nk,j/nk,0.]))
            es,ws = np.linalg.eigh(m)
            occ[(i,j)] = ws[:,es<0.]  # columns
    def U(a,b,mu):
        wa = occ[a]; wb = occ[b]
        d = np.linalg.det(np.conjugate(wa.T)@wb)
        return d/abs(d)
    tot = 0.
    for i in range(nk):
        for j in range(nk):
            a=(i,j); b=((i+1)%nk,j); c=((i+1)%nk,(j+1)%nk); d=(i,(j+1)%nk)
            F = np.log(U(a,b,0)*U(b,c,1)/U(d,c,0)/U(a,d,1))
            tot += F.imag
    return tot/(2*np.pi)

h = haldane(0.1,0.2)
print("FHS chern (mine, nk=24)     :", fhs_chern(h,24))
print("pyqula topology.chern nk=14 :", topology.chern(h,nk=14))
print("pyqula h.get_chern()        :", h.get_chern(nk=14))

# pointwise: berry_curvature (Wilson) vs RMP Kubo
for k in ([0.17,0.41],[0.05,0.9],[0.28,0.30]):
    es,om = kubo_from_scratch(h,k)
    kubo = np.sum(om[es<=0.])
    wil = topology.berry_curvature(h,np.array(k),dk=1e-3)
    print(f"k={k} kubo(RMP)={kubo: .6f}  topology.berry_curvature={wil: .6f}  ratio={wil/kubo: .6f}")
```

## `s3_calibrate_kubo_sign.py`

```python
"""Calibrate MY Kubo sign against the spin-1/2 monopole: for H = n(th,ph).sigma,
the UPPER band's Berry curvature integrates to -2*pi over the sphere in the
RMP convention A = i<u|grad u> (this is the Provost-Vallee value the repo's
own berry_curvature docstring cites)."""
import numpy as np

sx=np.array([[0,1],[1,0]],dtype=complex)
sy=np.array([[0,-1j],[1j,0]],dtype=complex)
sz=np.array([[1,0],[0,-1]],dtype=complex)

def H(th,ph):
    n=np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
    return n[0]*sx+n[1]*sy+n[2]*sz

def kubo(th,ph,d=1e-5):
    vx=(H(th+d,ph)-H(th-d,ph))/(2*d)
    vy=(H(th,ph+d)-H(th,ph-d))/(2*d)
    es,ws=np.linalg.eigh(H(th,ph)); ws=ws.T
    V1=np.conjugate(ws)@vx@ws.T; V2=np.conjugate(ws)@vy@ws.T
    om=np.zeros(2)
    for i in range(2):
        s=0.
        for j in range(2):
            if i==j: continue
            s+=np.imag(V1[i,j]*V2[j,i])/(es[i]-es[j])**2
        om[i]=-2*s
    return es,om

nth,nph=200,200
tot=np.zeros(2)
for i in range(nth):
    th=(i+0.5)*np.pi/nth
    for j in range(nph):
        ph=(j+0.5)*2*np.pi/nph
        es,om=kubo(th,ph)
        tot+=om*(np.pi/nth)*(2*np.pi/nph)
print("integral of my-Kubo curvature over the sphere, [lower,upper] =",tot)
print("expected RMP: lower=+2pi=%.4f  upper=-2pi=%.4f"%(2*np.pi,-2*np.pi))
```

## `s4_sign_convention.py`

Backs: **L4 topology** — berry_curvature / berry_phase document the OPPOSITE sign convention to what they return; the stated derivation is falsified by uij's double conjugation

```python
"""FINDING: topology.berry_curvature / berry_phase document a sign convention
they do not have.

Their SIGN CONVENTION docstrings (commit b3d1667) claim the returned value is
MINUS the Berry curvature / Berry phase of Xiao-Chang-Niu RMP 82, 1959 (2010)
(A = i<u|grad_k u>).  The derivation given is:

    "uij(a,b)[i,j] = <a_i|b_j> ... that product equals exp(-i * closed
     integral of A), so its argument is minus the Berry phase"

The premise is false.  occstates.occupied_states already returns CONJUGATED
wavefunctions (`wfs = np.conjugate(wfs.transpose())`), and overlap.uij
conjugates its first argument AGAIN, so

    uij(a,b)[i,j] = <b_j | a_i>,   not  <a_i | b_j>,

i.e. the elementwise complex conjugate of the intended link variable.  That
conjugates the closed link product, and the sign flips back: pyqula returns
+Omega_RMP, not -Omega_RMP.

Oracle: (1) an independent Kubo evaluation, calibrated to the RMP convention
on the spin-1/2 monopole (upper band integrates to -2*pi, the Provost-Vallee
value the docstring itself cites); (2) the identity uij(a,b) = <b|a>.
"""
import numpy as np
from pyqula import geometry, topology
from pyqula.topologytk.overlap import uij
from pyqula.topologytk.occstates import occupied_states

# ---- (0) uij returns <b_j|a_i>, not <a_i|b_j> --------------------------
g = geometry.honeycomb_lattice(); h = g.get_hamiltonian(has_spin=False)
h.add_haldane(0.1); h.add_sublattice_imbalance(0.2)
hk = h.get_hk_gen()
ka, kb = np.array([0.13,0.27,0.]), np.array([0.14,0.28,0.])
ea,wa = np.linalg.eigh(np.array(hk(ka).todense()) if hasattr(hk(ka),'todense') else np.array(hk(ka)))
eb,wb = np.linalg.eigh(np.array(hk(kb).todense()) if hasattr(hk(kb),'todense') else np.array(hk(kb)))
A = occupied_states(hk,ka); B = occupied_states(hk,kb)
U = uij(A,B)
ua = wa[:,ea<0.]; ub = wb[:,eb<0.]
naive = np.conjugate(ua.T)@ub          # <a_i|b_j>
print("uij(A,B)                =",U.ravel())
print("<a_i|b_j> (naive)       =",naive.ravel())
print("<b_j|a_i> = conj(naive) =",np.conjugate(naive).ravel())
print("uij == conj(<a|b>)?", np.allclose(U,np.conjugate(naive)))

# ---- (1) RMP-calibrated Kubo ------------------------------------------
def dense(m): return np.array(m.todense()) if hasattr(m,'todense') else np.array(m)
def kubo_rmp(h,k,dk=1e-5):
    hk = h.get_hk_gen(); k = np.array(list(k)+[0.]*(3-len(k)))
    dx=np.array([dk,0,0.]); dy=np.array([0,dk,0.])
    vx=(dense(hk(k+dx))-dense(hk(k-dx)))/(2*dk)
    vy=(dense(hk(k+dy))-dense(hk(k-dy)))/(2*dk)
    es,ws=np.linalg.eigh(dense(hk(k))); ws=ws.T
    V1=np.conjugate(ws)@vx@ws.T; V2=np.conjugate(ws)@vy@ws.T
    om=np.zeros(len(es))
    for i in range(len(es)):
        s=0.
        for j in range(len(es)):
            if i!=j: s+=np.imag(V1[i,j]*V2[j,i])/(es[i]-es[j])**2
        om[i]=-2*s
    return np.sum(om[es<=0.])
print()
print(" k                 Kubo(RMP)      berry_curvature   ratio")
for k in ([0.17,0.41],[0.05,0.90],[0.28,0.30],[0.44,0.11]):
    kb_=kubo_rmp(h,k); w=topology.berry_curvature(h,np.array(k),dk=1e-4)
    print(f" {k}   {kb_: 12.6f}   {w: 12.6f}   {w/kb_: .6f}")
print()
print("h.get_chern(nk=14) =",h.get_chern(nk=14),
      "  (Kubo/RMP BZ integral is +1, so pyqula = +Omega_RMP, not -Omega_RMP)")
```

## `s5_real_space_chern.py`

Backs: **L4 topology** — real_space_chern's only test asserts the trace of a commutator, which is zero by identity -- it passes for a trivial island as readily as a topological one

```python
"""topologytk/realspace.real_space_chern: its only test asserts a quantity
that is ZERO BY AN ALGEBRAIC IDENTITY (the trace of a commutator), so it
cannot discriminate a topological island from a trivial one."""
import numpy as np
from pyqula import islands, topology
import os, tempfile
os.chdir(tempfile.mkdtemp())

for t2 in [0.1, 0.0, -0.1, 0.3]:
    g = islands.get_geometry(name="honeycomb", n=6, nedges=4, rot=0.0, clean=False)
    h = g.get_hamiltonian(has_spin=False)
    if t2!=0.: h.add_haldane(t2)
    (r,c) = topology.real_space_chern(h)
    r = np.array(r)
    d2 = np.array([ri.dot(ri) for ri in r])
    core = d2 < np.max(d2)/9.   # deep bulk
    print(f"t2={t2:+.2f}  sum(c)={np.sum(c): .3e}   "
          f"mean bulk marker={np.mean(np.array(c)[core]): .4f}  (nsites_core={core.sum()})")
print()
print("the test asserts np.isclose(np.sum(c), 1.07e-14, atol=1e-6) -- "
      "identical for every t2, topological or not")
```

## `s6_spin_decomposition.py`

Backs: **L4 topology** — CLEARED: operator_berry's value correctness -- the lens's named target

```python
"""Oracle: sz-projected operator_berry vs an explicit per-spin-block
decomposition, on a Kane-Mele model where sz commutes with H."""
import numpy as np
from pyqula import geometry, topology, operators

def dense(m): return np.array(m.todense()) if hasattr(m,'todense') else np.array(m)

def km(soc=0.1,mass=0.0,rashba=0.0):
    g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=True)
    h.add_kane_mele(soc)
    if mass!=0.: h.add_sublattice_imbalance(mass)
    if rashba!=0.: h.add_rashba(rashba)
    return h

h=km(0.1,0.2)
sz=dense(operators.get_sz(h))
print("sz eigenvalues:",np.unique(np.round(np.diag(sz).real,6)))
hk=h.get_hk_gen()
k=np.array([0.17,0.41,0.])
m=dense(hk(k))
print("||[H(k),sz]|| =",np.linalg.norm(m@sz-sz@m))

def kubo_bands(mat_h,vx,vy,O=None):
    es,ws=np.linalg.eigh(mat_h); ws=ws.T
    vxo = (O@vx+vx@O)/2. if O is not None else vx
    V1=np.conjugate(ws)@vxo@ws.T; V2=np.conjugate(ws)@vy@ws.T
    n=len(es); om=np.zeros(n)
    for i in range(n):
        s=0.
        for j in range(n):
            if i!=j: s+=np.imag(V1[i,j]*V2[j,i])/(es[i]-es[j])**2
        om[i]=-2*s
    return es,om,ws

def vels(h,k,dk=1e-5):
    hk=h.get_hk_gen(); k=np.array(list(k)+[0.]*(3-len(k)))
    dx=np.array([dk,0,0.]); dy=np.array([0,dk,0.])
    return (dense(hk(k+dx))-dense(hk(k-dx)))/(2*dk),(dense(hk(k+dy))-dense(hk(k-dy)))/(2*dk)

print("\n k                Om_sz(pyqula)  sum_n sz_n Om_n   Om_up-Om_dn   Om_tot   Om_up+Om_dn")
for k in ([0.17,0.41],[0.31,0.33],[0.05,0.9]):
    vx,vy=vels(h,k)
    mk=dense(hk(np.array(list(k)+[0.])))
    es,om,ws=kubo_bands(mk,vx,vy)
    szn=np.array([np.real(np.conjugate(w)@sz@w) for w in ws])
    occ=es<=0.
    # explicit spin blocks: project H, vx, vy on the sz=+1 / -1 subspaces
    idx_up=np.where(np.diag(sz).real>0)[0]; idx_dn=np.where(np.diag(sz).real<0)[0]
    def blk(M,i): return M[np.ix_(i,i)]
    esu,omu,_=kubo_bands(blk(mk,idx_up),blk(vx,idx_up),blk(vy,idx_up))
    esd,omd,_=kubo_bands(blk(mk,idx_dn),blk(vx,idx_dn),blk(vy,idx_dn))
    Ou=np.sum(omu[esu<=0.]); Od=np.sum(omd[esd<=0.])
    pyq=topology.operator_berry(h,k=k,operator=operators.get_sz(h))
    pyq_tot=topology.operator_berry(h,k=k)
    print(f" {k}  {pyq: 13.6f}  {np.sum(szn[occ]*om[occ]): 14.6f}  {Ou-Od: 12.6f}  {pyq_tot: 9.4f} {Ou+Od: 11.6f}")

# integrated
nk=30
ks=[[(i+.5)/nk,(j+.5)/nk] for i in range(nk) for j in range(nk)]
cu=0.;cd=0.
for k in ks:
    vx,vy=vels(h,k); mk=dense(hk(np.array(list(k)+[0.])))
    idx_up=np.where(np.diag(sz).real>0)[0]; idx_dn=np.where(np.diag(sz).real<0)[0]
    def blk(M,i): return M[np.ix_(i,i)]
    esu,omu,_=kubo_bands(blk(mk,idx_up),blk(vx,idx_up),blk(vy,idx_up))
    esd,omd,_=kubo_bands(blk(mk,idx_dn),blk(vx,idx_dn),blk(vy,idx_dn))
    cu+=np.sum(omu[esu<=0.]); cd+=np.sum(omd[esd<=0.])
cu/=len(ks)*2*np.pi; cd/=len(ks)*2*np.pi
print(f"\nC_up={cu:.4f} C_dn={cd:.4f}  C_up-C_dn={cu-cd:.4f}  C_up+C_dn={cu+cd:.4f}")
print("topology.spin_chern(nk=24) =",topology.spin_chern(h,nk=24))
print("h.get_chern(nk=14)         =",h.get_chern(nk=14))
```

## `s7_operator_object.py`

Backs: **L4 topology** — operator_berry raises on an operators.Operator -- the exact type the code comment, commit message and test docstring all claim it supports

```python
"""Does operator_berry accept an operators.Operator (the type h.get_operator
returns and that operatorberry.py's comment says it must support)?"""
import numpy as np
from pyqula import geometry, topology, operators

def dense(m): return np.array(m.todense()) if hasattr(m,'todense') else np.array(m)
g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=True)
h.add_kane_mele(0.1); h.add_sublattice_imbalance(0.2)
k=[0.17,0.41]
for name in ["sz","sx","sublattice"]:
    op=h.get_operator(name)
    print("---",name,type(op).__name__)
    try:
        m=dense(op.get_matrix())
        b1=topology.operator_berry(h,k=k,operator=m)
        print("   raw matrix ->",b1)
    except Exception as e:
        print("   raw matrix EXC",type(e).__name__,e)
    try:
        b2=topology.operator_berry(h,k=k,operator=op)
        print("   Operator   ->",b2)
    except Exception as e:
        print("   Operator   EXC",type(e).__name__,repr(e)[:200])
# and a k-dependent operator (valley)
op=h.get_operator("valley")
print("--- valley",type(op).__name__)
try:
    print("   Operator ->",topology.operator_berry(h,k=k,operator=op))
except Exception as e:
    print("   Operator EXC",type(e).__name__,repr(e)[:300])
```

## `s8_gauge_qgt_supercell.py`

Backs: **L4 topology** — h.get_supercell([2,2]) dies with a bare IndexError while g.get_supercell([2,2]) accepts the same argument

Backs: **L4 topology** — CLEARED: operator_berry's b*pi*pi*8 vs multicell.derivative's missing 2*pi -- do they still cancel exactly?

```python
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology, operators, gauge
from pyqula.topologytk.qgt import quantum_geometric_tensor_k, berry_curvature_from_qgt

def haldane(t2=0.1,mass=0.2,spin=False):
    g=geometry.honeycomb_lattice(); h=g.get_hamiltonian(has_spin=spin)
    h.add_haldane(t2)
    if mass!=0.: h.add_sublattice_imbalance(mass)
    return h

h=haldane()
print("== A. operator_berry vs QGT Berry curvature, pointwise ==")
for k in ([0.17,0.41],[0.28,0.30],[0.05,0.9]):
    ob=topology.operator_berry(h,k=k)
    Q=quantum_geometric_tensor_k(h,k=list(k)+[0.])
    om=berry_curvature_from_qgt(Q)[0,1]
    print(f"  k={k}  operator_berry={ob: .6f}  QGT Omega_xy={om: .6f}  ratio={ob/om: .8f}")

print("\n== B. site-local gauge invariance (H -> U H U^dag, U=diag(e^{2pi i phi})) ==")
np.random.seed(0)
phis=np.random.random(len(h.geometry.r))
hg=gauge.hamiltonian_gauge_transformation(h,phis)
for k in ([0.17,0.41],[0.28,0.30]):
    print(f"  k={k}  plain={topology.operator_berry(h,k=k): .8f}  gauged={topology.operator_berry(hg,k=k): .8f}")
print("  chern plain",topology.chern(h,nk=14)," gauged",topology.chern(hg,nk=14))

print("\n== C. k-mesh origin offset (mesh_chern with an offset kmesh) ==")
from pyqula import klist
for off in [0.0,0.1,0.37]:
    nk=16
    km=np.array([[ (i+off)/nk,(j+off)/nk,0.] for i in range(nk) for j in range(nk)])
    print(f"  offset={off}  mesh_chern(kmesh)={topology.mesh_chern(h,kmesh=km)}")
print("  default klist.kmesh first points:",klist.kmesh(2,nk=4)[:3])

print("\n== D. supercell invariance ==")
h2=h.get_supercell([2,2])
print("  chern original     ",topology.chern(h,nk=14))
try:
    print("  chern 2x2 supercell",topology.chern(h2,nk=7))
except Exception as e:
    print("  chern supercell EXC",type(e).__name__,repr(e)[:200])
try:
    nk=12
    vals=[topology.operator_berry(h2,k=[(i+.5)/nk,(j+.5)/nk]) for i in range(nk) for j in range(nk)]
    print("  operator_berry BZ-avg/2pi on 2x2 supercell:",np.mean(vals)/(2*np.pi))
except Exception as e:
    print("  operator_berry supercell EXC",type(e).__name__,repr(e)[:300])

print("\n== E. Kane-Mele Z2 ==")
g=geometry.honeycomb_lattice(); hkm=g.get_hamiltonian(has_spin=True); hkm.add_kane_mele(0.1)
print("  z2 (soc only)     :",topology.z2_invariant(hkm,nk=20,nt=20))
hkm2=hkm.copy(); hkm2.add_sublattice_imbalance(1.5)
print("  z2 (large mass)   :",topology.z2_invariant(hkm2,nk=20,nt=20))
```

## `s9_sparse_crash.py`

Backs: **L4 topology** — operator_berry / operator_berry_bands / spin_chern raise ValueError on ANY sparse Hamiltonian, including every h.get_supercell(...)

```python
"""FINDING: topology.operator_berry / operator_berry_bands / spin_chern /
precise_spin_chern / write_spin_berry / bandstructure.berry_bands all raise
ValueError on ANY sparse Hamiltonian (h.turn_sparse(), and every
h.get_supercell(...), which sets is_sparse=True unconditionally).

Structural cause: topologytk/operatorberry.py:22-25 coerces with np.asarray.
multicell.derivative returns a scipy sparse matrix when h.is_sparse, and
np.asarray(<scipy sparse>) is a 0-d OBJECT array, not a dense 2d array, so the
very next line `operator@dhdx` raises
  "matmul: Input operand 1 does not have enough dimensions (has 0 ...)".
The package's own algebra.todense() handles both np.matrix and sparse; asarray
handles only np.matrix. This is a regression from f50d0db, which introduced the
asarray coercion to fix the np.matrix bug: the pre-f50d0db line
`(operator@dhdx + dhdx@operator)/2.` worked on sparse input (scipy defines
__rmatmul__/__matmul__).
"""
import numpy as np, os, tempfile
os.chdir(tempfile.mkdtemp())
from pyqula import geometry, topology, multicell, algebra
import scipy.sparse as sp

g=geometry.honeycomb_lattice()
h=g.get_hamiltonian(has_spin=True); h.add_kane_mele(0.1); h.add_sublattice_imbalance(0.2)

print("dense h  : spin_chern(nk=6) =",topology.spin_chern(h,nk=6))

hs=h.copy(); hs.turn_sparse()
print("sparse h : is_sparse =",hs.is_sparse)
d=multicell.derivative(multicell.turn_multicell(hs),np.array([0.1,0.2,0.]),order=[1,0])
print("   type(multicell.derivative) =",type(d).__name__)
print("   np.asarray(...).shape      =",np.asarray(d).shape,"dtype",np.asarray(d).dtype,"   <-- 0-d object array")
print("   algebra.todense(...).shape =",np.shape(algebra.todense(d)))
for name,fn in [("operator_berry",lambda: topology.operator_berry(hs,k=[0.17,0.41])),
                ("operator_berry_bands",lambda: topology.operator_berry_bands(hs,k=[0.17,0.41])),
                ("spin_chern",lambda: topology.spin_chern(hs,nk=4))]:
    try: print("   %-22s ->"%name,fn())
    except Exception as e: print("   %-22s EXC"%name,type(e).__name__,str(e)[:90])

h2=h.get_supercell([2,2,1])
print("supercell h.get_supercell([2,2,1]).is_sparse =",h2.is_sparse)
try: print("   spin_chern ->",topology.spin_chern(h2,nk=4))
except Exception as e: print("   spin_chern EXC",type(e).__name__,str(e)[:90])

# the pre-f50d0db expression works on sparse:
I=np.identity(d.shape[0],dtype=np.complex128)
print("pre-f50d0db expression on sparse input works:",np.shape((I@d + d@I)/2.))
```

## `scan.py`

```python
import ast,os,sys
root="<repo root>/tests"
def src_of(node,lines): return "\n".join(lines[node.lineno-1:node.end_lineno])
for dp,_,fns in os.walk(root):
    for fn in sorted(fns):
        if not fn.startswith("test_") or not fn.endswith(".py"): continue
        p=os.path.join(dp,fn); s=open(p).read(); lines=s.split("\n")
        try: t=ast.parse(s)
        except Exception as e: print("PARSE FAIL",p,e); continue
        for node in ast.walk(t):
            if not (isinstance(node,(ast.FunctionDef,)) and node.name.startswith("test_")): continue
            body=src_of(node,lines)
            asserts=[n for n in ast.walk(node) if isinstance(n,ast.Assert)]
            rel=os.path.relpath(p,root)
            # no assert at all and no pytest.raises
            if not asserts and "raises" not in body:
                print(f"NOASSERT   {rel}:{node.lineno} {node.name}")
            # assertions only of finite/notnone shape
            if asserts:
                txt=" ".join(ast.unparse(a.test) for a in asserts)
                if all(any(k in ast.unparse(a.test) for k in ("isfinite","is not None","isnan","shape","len(",">0","> 0")) for a in asserts):
                    print(f"WEAKONLY   {rel}:{node.lineno} {node.name} :: {txt[:160]}")
            # assertion nested inside an if
            for a in asserts:
                pass
```

## `scan2.py`

Backs: **L5 siblings** — unfolding.unfolded_bands raises NameError on an undefined numfp before its NotImplementedError

```python
import ast,os,re
root="<repo root>/tests"
print("=== asserts nested inside an if / for (may never execute) ===")
for dp,_,fns in os.walk(root):
    if "qutecipy" in dp: continue
    for fn in sorted(fns):
        if not fn.startswith("test_") or not fn.endswith(".py"): continue
        p=os.path.join(dp,fn); s=open(p).read(); t=ast.parse(s)
        for n in ast.walk(t):
            if not (isinstance(n,ast.FunctionDef) and n.name.startswith("test_")): continue
            # find asserts whose only ancestor-path includes an If
            def walk(node,anc):
                for ch in ast.iter_child_nodes(node):
                    if isinstance(ch,ast.Assert) and any(isinstance(a,ast.If) for a in anc):
                        print(f"  {os.path.relpath(p,root)}:{ch.lineno} in {n.name}: "
                              f"{ast.unparse(ch.test)[:90]}")
                    walk(ch,anc+[node])
            walk(n,[])
print()
print("=== loose tolerances (atol/rtol >= 0.1, or a bare '< x' with x>=0.5) ===")
pat=re.compile(r"(atol|rtol)\s*=\s*([0-9.]+)")
for dp,_,fns in os.walk(root):
    if "qutecipy" in dp: continue
    for fn in sorted(fns):
        if not fn.startswith("test_") or not fn.endswith(".py"): continue
        p=os.path.join(dp,fn)
        for i,line in enumerate(open(p),1):
            for m in pat.finditer(line):
                try: v=float(m.group(2))
                except: continue
                if v>=0.1 and "assert" in line:
                    print(f"  {os.path.relpath(p,root)}:{i}: {line.strip()[:120]}")
```

## `scan_unused_args.py`

```python
import ast, os, sys

ROOT="<repo root>/src/pyqula"
FENCE=("scftk","meanfield.py","sctk","superconductivity.py","greentk","transporttk",
       "keldyshtk","aaatk","qtcitk","heterostructures.py","topology.py","topologytk",
       "multicell.py","gauge.py","qutecipytk")

def fenced(path):
    rel=os.path.relpath(path,ROOT)
    parts=rel.split(os.sep)
    for f in FENCE:
        if f.endswith(".py"):
            if rel==f: return True
        else:
            if parts[0]==f: return True
    return False

class NameCollector(ast.NodeVisitor):
    def __init__(self): self.names=set()
    def visit_Name(self,n): self.names.add(n.id); self.generic_visit(n)
    def visit_Attribute(self,n): self.names.add(n.attr); self.generic_visit(n)
    def visit_keyword(self,n):
        if n.arg: self.names.add(n.arg)
        self.generic_visit(n)
    def visit_arg(self,n): self.names.add(n.arg); self.generic_visit(n)
    def visit_Constant(self,n):
        if isinstance(n.value,str):
            for tok in n.value.replace('"',' ').replace("'",' ').split():
                self.names.add(tok.strip('=,()[]{}:'))
        self.generic_visit(n)

results=[]
for dirpath,dirs,files in os.walk(ROOT):
    for fn in files:
        if not fn.endswith(".py"): continue
        p=os.path.join(dirpath,fn)
        if fenced(p): continue
        try: tree=ast.parse(open(p).read())
        except Exception: continue
        for node in ast.walk(tree):
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
            a=node.args
            params=[x.arg for x in a.posonlyargs+a.args+a.kwonlyargs]
            if a.vararg: params.append(a.vararg.arg)
            has_kwargs = a.kwarg is not None
            # collect names used in the body only
            nc=NameCollector()
            for st in node.body: nc.visit(st)
            body_names=nc.names
            # also nested defaults of inner funcs etc are covered by generic visit
            unused=[p_ for p_ in params if p_ not in body_names and p_ not in ("self","cls")]
            if unused:
                results.append((os.path.relpath(p,ROOT),node.lineno,node.name,unused,has_kwargs))

for r in sorted(results):
    print(f"{r[0]}:{r[1]} {r[2]}  UNUSED={r[3]}  **kwargs={r[4]}")
print("TOTAL",len(results))
```

## `smoke.py`

Backs: **L5 siblings** — Hamiltonian.enforce_eh dies on a Python-2 absolute import before reaching its own NotImplementedError

```python
import numpy as np, pyqula, traceback
assert "/src/pyqula" in pyqula.__file__, pyqula.__file__
from pyqula import geometry
g = geometry.honeycomb_lattice()
h = g.get_hamiltonian(has_spin=True)

calls = [
 ("h.spinless2full", lambda: h.spinless2full(np.identity(2))),
 ("h.spinful2full", lambda: h.spinful2full(np.identity(4))),
 ("h.get_average_spin_splitting", lambda: h.get_average_spin_splitting()),
 ("h.get_gf", lambda: h.get_gf(e=0.1,delta=1e-2)),
 ("h.modify_hamiltonian_matrices", lambda: h.copy().modify_hamiltonian_matrices(lambda m: m)),
 ("h.remove_pairing", lambda: h.copy().remove_pairing()),
 ("h.remove_sites", lambda: h.copy().remove_sites([0])),
 ("h.full2profile", lambda: h.full2profile(np.random.random(h.intra.shape[0]))),
 ("h.get_quantum_geometric_tensor", lambda: h.get_quantum_geometric_tensor(k=[0.1,0.2,0.])),
 ("h.get_drude_weight", lambda: h.get_drude_weight(nk=4)),
 ("h.get_sum_rule_weight", lambda: h.get_sum_rule_weight(nk=4)),
 ("h.get_polarizability", lambda: h.get_polarizability(nk=3)),
 ("h.has_time_reversal_symmetry", lambda: h.has_time_reversal_symmetry()),
 ("h.get_rkky", lambda: h.get_rkky(nk=4)),
 ("h.to_canonical_gauge", lambda: h.copy().to_canonical_gauge()),
 ("h.print_hamiltonian", lambda: h.print_hamiltonian()),
 ("h.get_bandwidth", lambda: h.get_bandwidth()),
 ("h.diagonalize", lambda: h.diagonalize()),
 ("h.get_density", lambda: h.get_density(nk=4)),
 ("h.same_hamiltonian", lambda: h.same_hamiltonian(h.copy())),
 ("h.first_neighbors", lambda: h.copy().first_neighbors()),
 ("h.enforce_eh", lambda: h.copy().enforce_eh()),
 ("h.add_chiral_kekule", lambda: h.copy().add_chiral_kekule(t=0.1)),
 ("h.add_crystal_field", lambda: h.copy().add_crystal_field(0.1)),
 ("h.get_1dh", lambda: h.get_1dh(0)),
 ("h.get_no_multicell", lambda: h.get_no_multicell()),
 ("h.write_non_unitarity", lambda: h.write_non_unitarity()),
 ("h.get_ipr", lambda: h.get_ipr()),
 ("h.get_single_vev", lambda: h.get_single_vev("sz")),
 ("g.get_neighbor_distances", lambda: g.get_neighbor_distances()),
 ("g.normalize_nn_distance", lambda: g.copy().normalize_nn_distance()),
 ("g.get_default_kpath", lambda: g.get_default_kpath()),
 ("g.get_orthogonal", lambda: g.get_orthogonal()),
 ("g.get_closest_position", lambda: g.get_closest_position([0.1,0.1,0.])),
 ("g.xyz2r", lambda: g.copy().xyz2r()),
 ("g.write_positions", lambda: g.write_positions()),
 ("g.set_origin", lambda: g.copy().set_origin()),
 ("g.get_lattice_name", lambda: g.get_lattice_name()),
 ("g.get_diameter", lambda: g.get_diameter()),
 ("g.periodic_vector", lambda: g.periodic_vector()),
 ("g.neighbor_directions", lambda: g.neighbor_directions()),
 ("g.get_ncells", lambda: g.get_ncells()),
 ("g.center_in_atom", lambda: g.copy().center_in_atom()),
 ("g.update_reciprocal", lambda: g.copy().update_reciprocal()),
 ("g.K2k", lambda: g.K2k([0.1,0.2,0.])),
 ("g.fractional2real", lambda: g.copy().fractional2real()),
 ("g.real2fractional", lambda: g.copy().real2fractional()),
]
import os,contextlib,io
for name,f in calls:
    try:
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf): r=f()
        print("OK   ",name)
    except Exception as e:
        print("FAIL ",name,"->",type(e).__name__,":",str(e)[:160])
```

## `su2_kernel.py`

Backs: **L1 SCF/Nambu** — CLEARED: SU(2) covariance of the whole BdG mean-field kernel (get_mf_bdg) and of spinspin._rot_dm's conjugate-sandwiched rotation

```python
"""SU(2) equivariance of the BdG mean-field kernel.

n_i n_j (density-density, the vd matrix built by _build_density_v) is SU(2)
invariant, so its Hartree-Fock + anomalous decoupling must be EQUIVARIANT:
  get_mf_bdg(v, U dm U^dag) == U get_mf_bdg(v, dm) U^dag
for any global spin rotation U (Nambu-embedded).  Oracle: the symmetry itself.
"""
import numpy as np
from pyqula import geometry
from pyqula.rotate_spin import global_spin_rotation as gsr
from pyqula.scftk.superscf import get_mf_bdg
from pyqula.scftk.spinspin import _build_density_v, _build_v, _rot_dm, _rot_dict
from pyqula.rotate_spin import build_rotation_matrix

rng = np.random.default_rng(3)
g = geometry.chain()

def make_h(seed):
    r = np.random.default_rng(seed)
    h = g.get_hamiltonian(has_spin=True)
    h.turn_nambu()
    # break SU(2) and U(1): random exchange direction + pairing
    return h

def dm_of(h,ds,nk=8):
    return h.get_density_matrix(ds=ds,nk=nk)

# build the interaction (spin-orbital sized, NOT nambu) -- V1 + U
hspin = g.get_hamiltonian(has_spin=True)
v = _build_density_v(hspin, V1=1.0, V2=0.0, V3=0.0, U=-2.0)
ds = [(0,0,0)] + list(v.keys())

vec = rng.random(3)-0.5 ; vec/=np.linalg.norm(vec)
angle = 0.37

for label,setup in [("magnetic only", lambda h: h.add_exchange([0.13,-0.21,0.3])),
                    ("pairing only",  lambda h: h.add_swave(0.25)),
                    ("magnetic+pairing", lambda h: (h.add_exchange([0.13,-0.21,0.3]),h.add_swave(0.25)))]:
    h = g.get_hamiltonian(has_spin=True); h.turn_nambu(); setup(h)
    hr = h.copy()
    # rotate the whole Hamiltonian
    d = hr.get_dict()
    d = {k: gsr(np.array(m),vector=vec,angle=angle) for k,m in d.items()}
    hr.set_multihopping(__import__("pyqula.multihopping",fromlist=["MultiHopping"]).MultiHopping(d))

    dm  = dm_of(h,ds)
    dmr = dm_of(hr,ds)
    mf  = get_mf_bdg(v,dm)
    mfr = get_mf_bdg(v,dmr)
    # rotate mf the way a Hamiltonian-like object rotates
    mf_rot = {k: gsr(np.array(m),vector=vec,angle=angle) for k,m in mf.items()}
    err = max(np.max(np.abs(np.array(mf_rot[k])-np.array(mfr[k]))) for k in mf)
    scale = max(np.max(np.abs(np.array(mfr[k]))) for k in mf)
    print("%-18s  max|R mf R^dag - mf(R dm R^dag)| = %.3e   (scale %.3e)"%(label,err,scale))

    # also check that the density matrix itself rotates as _rot_dm claims
    R = build_rotation_matrix(1,vector=vec,angle=angle)
    dm_rot_pred = _rot_dm(dm,R)
    err2 = max(np.max(np.abs(np.array(dm_rot_pred[k])-np.array(dmr[k]))) for k in dm)
    sc2 = max(np.max(np.abs(np.array(dmr[k]))) for k in dm)
    print("   _rot_dm(dm) vs dm(rotated H):   %.3e   (scale %.3e)"%(err2,sc2))
```

## `t10_phs_modes.py`

```python
import numpy as np, warnings
warnings.filterwarnings("ignore")
from pyqula import geometry
modes = ["swave","extended_swave","triplet","pwave","nodal_fwave","chiral_pwave",
         "chiral_fwave","chiral_dwave","chiral_gwave","antihaldane","haldane",
         "swavez","px","dpid","swaveA","swaveB","swavesublattice","dx2y2",
         "nodal_dwave","dxy","snn","C3nn","SnnAB","deltaud"]
g = geometry.honeycomb_lattice()
k = np.array([0.137,0.291,0.])
for m in modes:
    try:
        h = g.get_hamiltonian()
        h.add_pairing(delta=0.3, mode=m, d=[0.,0.,1.])
    except Exception as e:
        print(f"{m:18s} BUILD-FAIL {type(e).__name__}: {str(e)[:60]}"); continue
    try:
        hk = h.get_hk_gen()
        m1 = np.array(hk(k)); m2 = np.array(hk(-k))
        herm = np.max(np.abs(m1-np.conjugate(m1.T)))
        e1 = np.sort(np.linalg.eigvalsh(m1)); e2 = np.sort(np.linalg.eigvalsh(m2))
        phs = np.max(np.abs(e1 + e2[::-1]))   # E_n(k) = -E_n(-k)
        anom = np.max(np.abs(m1)) # nonzero check
        from pyqula.superconductivity import get_eh_sector
        a01 = np.max(np.abs(get_eh_sector(m1,i=0,j=1)))
        print(f"{m:18s} herm={herm:.2e}  PHS_resid={phs:.2e}  |anom|={a01:.4f}")
    except Exception as e:
        print(f"{m:18s} EVAL-FAIL {type(e).__name__}: {str(e)[:70]}")
```

## `t11_identify.py`

```python
import numpy as np, traceback
from pyqula import geometry
from pyqula import superconductivity as sc
g=geometry.chain()
# spinless BdG with real pairing
h=g.get_hamiltonian(has_spin=False); h.add_swave(0.3)
print("spinless_nambu:",h.check_mode("spinless_nambu"))
for lbl,f in [("identify_superconductivity",lambda: sc.identify_superconductivity(h)),
              ("superconductivity_type",lambda: sc.superconductivity_type(h))]:
    try: print(lbl,"->",f())
    except Exception as e: print(lbl,"->",type(e).__name__+":",str(e)[:100])
# also via meanfield entry point
from pyqula import meanfield
```

## `t12_orderparam.py`

```python
import numpy as np
from pyqula import geometry
from pyqula.sctk.orderparameter import singlet, triplet
g=geometry.square_lattice()
for D in [0.3,0.6]:
    h=g.get_hamiltonian(); h.add_swave(D)
    hn=g.get_hamiltonian(); hn.setup_nambu_spinor()
    mf = h - hn
    print("swave  D=",D," singlet(mf)=",np.round(singlet(mf,nk=6),6)," triplet(mf)=",np.round(triplet(mf,nk=6),6),
          "  4*D^2*nsites=",4*D**2*len(g.r))
for D in [0.3]:
    h=g.get_hamiltonian(); h.add_pairing(delta=D,mode="triplet",d=[0,0,1.])
    hn=g.get_hamiltonian(); hn.setup_nambu_spinor()
    mf=h-hn
    print("triplet D=",D," singlet(mf)=",np.round(singlet(mf,nk=6),6)," triplet(mf)=",np.round(triplet(mf,nk=6),6))
```

## `t13_perf_extract.py`

```python
"""extract_triplet_pairing is an O(nsites^2) pure-python double loop and is
re-run at every k-point by dvector_non_unitarity / average_hamiltonian_dvector.
Compare it against the identical strided-slice expression."""
import numpy as np, time
import pyqula.superconductivity  # avoid the sctk.extract circular-import order trap
from pyqula.sctk.extract import extract_triplet_pairing

def vectorized(m):
    m = np.asarray(m)
    ud = (m[0::4,2::4] - np.conjugate(m[3::4,1::4].T))/2.
    dd = m[1::4,2::4]
    uu = m[0::4,3::4]
    return (uu,dd,ud)

for nr in [20,60,120]:
    m = (np.random.random((4*nr,4*nr))+1j*np.random.random((4*nr,4*nr)))
    a = extract_triplet_pairing(m); b = vectorized(m)
    same = max(np.max(np.abs(a[i]-b[i])) for i in range(3))
    t0=time.time()
    for _ in range(5): extract_triplet_pairing(m)
    t1=time.time()
    for _ in range(5): vectorized(m)
    t2=time.time()
    print(f"nsites={nr:4d}  identical={same:.1e}  loop={t1-t0:.4f}s  strided={t2-t1:.4f}s  ratio={(t1-t0)/(t2-t1):.0f}x")
```

## `t14_perf_dvector.py`

```python
import numpy as np, time
import pyqula.superconductivity
from pyqula import geometry
import pyqula.sctk.extract as ex
import pyqula.sctk.dvector as dv

orig = ex.extract_triplet_pairing
calls = [0]
def counted(m):
    calls[0] += 1
    return orig(m)
def vectorized(m):
    from pyqula import algebra
    m = np.asarray(algebra.todense(m))
    return (m[0::4,3::4], m[1::4,2::4], (m[0::4,2::4]-np.conjugate(m[3::4,1::4].T))/2.)

g = geometry.honeycomb_lattice().supercell(5)   # 50 sites
h = g.get_hamiltonian()
h.add_pairing(delta=0.3, mode="triplet", d=[1.,1j,0.])
print("sites",len(g.r),"matrix",h.intra.shape)

dv.extract = ex
ex.extract_triplet_pairing = counted
t0=time.time(); q1 = h.get_dvector_non_unitarity(nk=6); t1=time.time()
print("loop version    %.3f s, extract_triplet_pairing calls = %d"%(t1-t0,calls[0]))
ex.extract_triplet_pairing = vectorized
t2=time.time(); q2 = h.get_dvector_non_unitarity(nk=6); t3=time.time()
print("strided version %.3f s  ratio %.0fx"%(t3-t2,(t1-t0)/(t3-t2)))
print("max |difference| =", np.max(np.abs(q1-q2)))
```

## `t15_perf_einsum.py`

```python
"""sctk/superfluidweight.py:_superfluid_weight_at line ~481:
     np.einsum("ij,jk,ki->i", wsc.T, B, ws)
np.einsum with three operands and optimize=False (the default) builds one
naive nested loop -- no BLAS.  The same quantity is diag(ws^dag B ws)."""
import numpy as np, time
for n in [200,400,800]:
    ws = np.linalg.qr(np.random.random((n,n))+1j*np.random.random((n,n)))[0]
    B  = np.random.random((n,n))+1j*np.random.random((n,n))
    wsc = np.conjugate(ws)
    t0=time.time(); a = np.einsum("ij,jk,ki->i",wsc.T,B,ws); t1=time.time()
    b = np.sum(wsc*(B@ws),axis=0)
    t2=time.time(); b = np.sum(wsc*(B@ws),axis=0); t3=time.time()
    c = np.einsum("ij,jk,ki->i",wsc.T,B,ws,optimize=True)
    print(f"n={n:4d}  einsum(default)={t1-t0:.4f}s  gemm+sum={t3-t2:.4f}s  ratio={(t1-t0)/(t3-t2):.0f}x  maxdiff={np.max(np.abs(a-b)):.2e}")
```

## `t16_perf_sfw_e2e.py`

```python
import numpy as np, time
from pyqula import geometry
import pyqula.sctk.superfluidweight as sfw
from pyqula.sctk.superfluidweight import _divided_difference, _fermi

def patched(es,ws,A,B,T,nd):
    W = _divided_difference(es,es,T); wsc = np.conjugate(ws)
    Arot = [wsc.T@a@ws for a in A]
    para = np.zeros((nd,nd))
    for a in range(nd):
        for b in range(a,nd):
            v = np.sum(W*Arot[a]*Arot[b].T).real; para[a,b]=v; para[b,a]=v
    nf = _fermi(es,T); dia = np.zeros((nd,nd))
    for (a,b) in B:
        if a>b: continue
        v = np.sum(nf*np.sum(wsc*(B[(a,b)]@ws),axis=0)).real   # gemm form
        dia[a,b]=v; dia[b,a]=v
    return para,dia

g = geometry.honeycomb_lattice().supercell(4)   # 32 sites -> 128x128 BdG
h = g.get_hamiltonian(); h.add_onsite(-0.4); h.add_swave(0.3)
print("BdG matrix",h.intra.shape)
orig = sfw._superfluid_weight_at
t0=time.time(); D1 = h.get_superfluid_weight(nk=6); t1=time.time()
sfw._superfluid_weight_at = patched
t2=time.time(); D2 = h.get_superfluid_weight(nk=6); t3=time.time()
sfw._superfluid_weight_at = orig
print("as shipped  %.3f s"%(t1-t0))
print("gemm form   %.3f s   ratio %.1fx"%(t3-t2,(t1-t0)/(t3-t2)))
print("max |D1-D2| =",np.max(np.abs(D1-D2)), " D =",np.round(D1[0,0],8))
```

## `t17_abs_spatial.py`

```python
import numpy as np
from pyqula import geometry
g = geometry.honeycomb_lattice()
# site-dependent s-wave gap: 0.3 on A, 0.1 on B
f = lambda r: 0.3 if r[0]<0 else 0.1
h = g.get_hamiltonian(); h.add_swave(f)
print("true on-site gaps (extract 'swave'):", np.round(np.abs(h.extract("swave")),6))
print("absolute_spatial_delta            :", np.round(h.extract("absolute_spatial_delta",nk=8),6))
print("ratio                             :",
      np.round(h.extract("absolute_spatial_delta",nk=8)/np.abs(h.extract("swave")),6), " (sqrt2 =",round(np.sqrt(2),6),")")
print("absolute_delta (mean, for scale)  :", np.round(np.real(h.extract("absolute_delta",nk=8)),6),
      " sqrt(mean |D|^2) =",round(np.sqrt((0.3**2+0.1**2)/2),6))
```

## `t3_nambu_spinless.py`

```python
import numpy as np
from pyqula import geometry
g = geometry.chain()
for name in ["turn_nambu","setup_nambu_spinor"]:
    h = g.get_hamiltonian(has_spin=False)
    print("before",name,": has_spin",h.has_spin,"has_eh",h.has_eh,"dim",h.intra.shape)
    getattr(h,name)()
    print("after ",name,": has_spin",h.has_spin,"has_eh",h.has_eh,"dim",h.intra.shape,
          "mode spinless_nambu",h.check_mode("spinless_nambu"),
          "mode spinful_nambu",h.check_mode("spinful_nambu"))
# and via add_pairing / add_swave
h1 = g.get_hamiltonian(has_spin=False); h1.add_swave(0.2)
h2 = g.get_hamiltonian(has_spin=False); h2.add_pairing(delta=0.2,mode="swave")
print("add_swave   on spinless:",h1.intra.shape, h1.has_spin, h1.has_eh)
print("add_pairing on spinless:",h2.intra.shape, h2.has_spin, h2.has_eh)
```

## `t4_pairing_bloch.py`

```python
import numpy as np
from pyqula import geometry
from pyqula.superconductivity import get_eh_sector
from pyqula.sctk.extract import extract_singlet_pairing, extract_triplet_pairing

D = 0.3
g = geometry.square_lattice()
h = g.get_hamiltonian()
h.add_pairing(delta=D, mode="extended_swave")
hk = h.get_hk_gen()
for k in [[0.13,0.27,0.],[0.4,-0.1,0.]]:
    m = hk(k)
    an = get_eh_sector(np.array(m),i=0,j=1)   # anomalous block, 2N x 2N (spin)
    kx,ky = k[0],k[1]
    expect = 2*D*(np.cos(2*np.pi*kx)+np.cos(2*np.pi*ky))
    print("k",k,"anomalous diag[0]",np.round(an[0,0],6),
          " 2D(cos+cos)=",np.round(expect,6),
          " ratio",np.round(an[0,0]/expect,6))
    print("   offdiag max", np.round(np.max(np.abs(an-np.diag(np.diag(an)))),8))
    ud = extract_singlet_pairing(np.array(m))
    uu,dd,udt = extract_triplet_pairing(np.array(m))
    print("   singlet ud[0,0]",np.round(ud[0,0],6)," triplet uu,dd,ud max",
          np.round(np.max(np.abs(uu)),8),np.round(np.max(np.abs(dd)),8),np.round(np.max(np.abs(udt)),8))
```

## `t5_dvector.py`

```python
import numpy as np
from pyqula import geometry
g = geometry.square_lattice()

# s-wave: d-vector must vanish
h = g.get_hamiltonian(); h.add_swave(0.3)
print("swave  avg dvector (|dx|^2,|dy|^2,|dz|^2):", np.round(h.get_average_dvector(nk=6),8))
print("swave  nonunit:", np.round(h.get_average_dvector(nk=6,non_unitarity=True),8))

# triplet along each axis
for d in [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]:
    h = g.get_hamiltonian()
    h.add_pairing(delta=0.3, mode="triplet", d=d)
    av = h.get_average_dvector(nk=6)
    print("triplet d=",d," avg |d|^2 =",np.round(av,6), " normalized",np.round(av/np.sum(av),4))

# non-unitarity: pure up-up pairing d ~ (1,i,0) -> q along +z
h = g.get_hamiltonian()
h.add_pairing(delta=0.3, mode="triplet", d=[1.,1j,0.])
q = h.get_dvector_non_unitarity(nk=6)
print("d=(1,i,0) -> q per site:", np.round(q,6))
```

## `t6_absolute.py`

```python
import numpy as np
from pyqula import geometry
print("Oracle: for a uniform on-site s-wave gap Delta, both routines claim to")
print("return 'the absolute value of the SC order'.")
for D in [0.3,0.5]:
    for lat,name in [(geometry.square_lattice(),"square"),(geometry.honeycomb_lattice(),"honeycomb"),(geometry.chain(),"chain")]:
        h = lat.get_hamiltonian(); h.add_swave(D)
        a = h.extract("absolute_delta",nk=8)
        b = h.extract("absolute_spatial_delta",nk=8)
        print(f"{name:10s} Delta={D}  absolute_delta={np.round(np.real(a),6)}  absolute_spatial_delta={np.round(b,6)}")
```

## `t7_spinless_nambu_remove.py`

```python
import numpy as np, traceback
from pyqula import geometry
g = geometry.chain()
h = g.get_hamiltonian(has_spin=False)
h.add_swave(0.3)           # spinless BdG, the mode pyqula itself calls "spinless_nambu"
print("mode spinless_nambu:",h.check_mode("spinless_nambu"),"has_spin",h.has_spin,"has_eh",h.has_eh)
for label,f in [("h.remove_nambu()",lambda: h.copy().remove_nambu()),
                ("h.get_anomalous_hamiltonian()",lambda: h.get_anomalous_hamiltonian()),
                ("h.extract('absolute_delta')",lambda: h.extract("absolute_delta",nk=4)),
                ("h.extract('absolute_spatial_delta')",lambda: h.extract("absolute_spatial_delta",nk=4)),
                ("h.extract('deltak')",lambda: h.extract("deltak",nk=4))]:
    try:
        r = f(); print(label,"-> ok",type(r).__name__)
    except Exception as e:
        print(label,"->",type(e).__name__+":",str(e)[:90])
# reach into transport: is_sc test used to decide if a lead is superconducting
from pyqula.transporttk.didv import generic_didv
import pyqula.transporttk.didv as didv
try:
    print("didv.is_sc(h) ->", didv.is_sc(h))
except Exception as e:
    print("didv.is_sc(h) ->",type(e).__name__+":",str(e)[:90])
```

## `t8_transport_spinless_nambu.py`

```python
import numpy as np, traceback
from pyqula import geometry, heterostructures
g = geometry.chain()
h1 = g.get_hamiltonian(has_spin=False); h1.add_swave(0.2)
h2 = g.get_hamiltonian(has_spin=False); h2.add_swave(0.2)
print("leads spinless_nambu:",h1.check_mode("spinless_nambu"))
ht = heterostructures.build(h1,h2)
try:
    print("didv:",ht.didv(energy=0.05))
except Exception:
    traceback.print_exc()
```

## `t9_misc.py`

```python
import numpy as np
from pyqula import geometry
g=geometry.square_lattice()

print("=== singlet operator on wrong Hilbert spaces ===")
for lbl,mk in [("spinful non-Nambu",lambda: g.get_hamiltonian()),
               ("spinless",lambda: g.get_hamiltonian(has_spin=False)),
               ("spinful_nambu",lambda: (lambda h:(h.add_swave(0.3),h)[1])(g.get_hamiltonian()))]:
    h=mk()
    try:
        op=h.get_operator("singlet")
        print(lbl,"op built; h.intra",h.intra.shape)
        try:
            print("   bands with op:",np.shape(h.get_bands(operator=op,nk=3)))
        except Exception as e: print("   apply ->",type(e).__name__+":",str(e)[:80])
    except Exception as e:
        print(lbl,"-> ",type(e).__name__+":",str(e)[:90])

print()
print("=== superfluid weight: spinless_nambu vs spinful_nambu (docstring claims factor 2) ===")
for D in [0.2,0.4]:
    hs=g.get_hamiltonian(); hs.add_onsite(-0.6); hs.add_swave(D)
    hl=g.get_hamiltonian(has_spin=False); hl.add_onsite(-0.6); hl.add_swave(D)
    Ds=hs.get_superfluid_weight(nk=16); Dl=hl.get_superfluid_weight(nk=16)
    print("D=",D," spinful",np.round(Ds[0,0],8)," spinless",np.round(Dl[0,0],8),
          " ratio",np.round(Ds[0,0]/Dl[0,0],8))
```

## `toplevel.py`

```python
import ast,os
R="<repo root>"
def blob(d):
    s=[]
    for dp,_,fns in os.walk(R+"/"+d):
        for f in fns:
            if f.endswith(".py"): s.append(open(os.path.join(dp,f),errors="ignore").read())
    return "\n".join(s)
T=blob("tests"); E=blob("examples")
out=[]
for f in sorted(os.listdir(R+"/src/pyqula")):
    if not f.endswith(".py") or f=="__init__.py": continue
    p=R+"/src/pyqula/"+f
    try: t=ast.parse(open(p).read())
    except: continue
    for n in t.body:
        if isinstance(n,ast.FunctionDef) and not n.name.startswith("_"):
            nm=n.name
            if len(nm)<5: continue
            if T.count(nm)==0:
                out.append((f,nm,n.lineno,E.count(nm)))
print(f"{len(out)} top-level public functions with ZERO occurrences in tests/")
byf={}
for f,nm,l,e in out: byf.setdefault(f,[]).append((nm,l,e))
for f in sorted(byf,key=lambda x:-len(byf[x]))[:25]:
    ex=[f"{nm}{'*' if e else ''}" for nm,l,e in byf[f]]
    print(f"  {f:28s} ({len(ex):2d}) {', '.join(ex[:9])}")
print("  (* = appears in examples/)")
```

## `unusedargs.py`

```python
import ast,os
R="<repo root>/src/pyqula"
INTEREST={"operator","delta","nk","energies","kpath","filling","mode","nsuper",
          "temperature","T","num_bands","ewindow","nrep","callback","solver","channel","write"}
out=[]
for root,d,fs in os.walk(R):
    if "qutecipytk" in root: continue
    for f in fs:
        if not f.endswith(".py"): continue
        p=os.path.join(root,f)
        try: t=ast.parse(open(p).read())
        except Exception: continue
        for n in ast.walk(t):
            if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
            a=n.args
            names=[x.arg for x in a.posonlyargs+a.args+a.kwonlyargs]
            haskw = a.kwarg is not None
            used=set()
            for s in ast.walk(n):
                if isinstance(s,ast.Name): used.add(s.id)
                if isinstance(s,ast.arg): pass
                if isinstance(s,ast.keyword) and s.arg: pass
            # also names used as **kwargs forwarding
            if haskw: used.add(a.kwarg)
            kwused = a.kwarg in used if haskw else True
            # names appearing as keyword=... values are Name nodes already
            for nm in names:
                if nm in ("self","cls"): continue
                if nm not in INTEREST: continue
                if nm not in used:
                    out.append((os.path.relpath(p,R),n.lineno,n.name,nm,"unused param"))
            if haskw and a.kwarg not in used:
                out.append((os.path.relpath(p,R),n.lineno,n.name,"**"+a.kwarg,"kwargs sink"))
for o in out:
    if o[4]=="unused param": print(o)
print("---- kwargs sinks:",sum(1 for o in out if o[4]=="kwargs sink"))
```

## `vj_vs_v.py`

Backs: **L1 SCF/Nambu** — CLEARED: VJinteraction vs Vinteraction disagreeing for a U-only interaction on a Nambu Hamiltonian (only V1 is covered by tests/scf/test_spinspin_nambu.py)

```python
"""VJinteraction with ONLY U on a Nambu Hamiltonian vs Vinteraction with the
same U -- the same interaction through two independent SCF loops."""
import numpy as np, os
from pyqula import geometry, meanfield
from pyqula.superconductivity import get_eh_sector

def mkh():
    g = geometry.chain(); h = g.get_hamiltonian(); h.setup_nambu_spinor(); return h

for U in (-2.0,-3.0):
  for gm in (0.2,):
    if os.path.exists("MF.pkl"): os.remove("MF.pkl")
    h = mkh(); h0=h.copy(); guess = h.copy(); guess.add_swave(gm)
    sV = meanfield.Vinteraction(h,U=U,filling=0.5,nk=20,mf=guess,mix=0.3,
            maxerror=1e-8,maxite=2000,load_mf=False,verbose=0)
    h = mkh(); guess = h.copy(); guess.add_swave(gm)
    sJ = meanfield.VJinteraction(h,U=U,filling=0.5,nk=20,mf=guess,mix=0.3,
            maxerror=1e-8,maxite=2000,verbose=0)
    for nm,s in (("Vinteraction",sV),("VJinteraction",sJ)):
        mf = np.array(s.hamiltonian.intra)-np.array(h0.intra)
        eh = get_eh_sector(mf,i=0,j=1); ee = get_eh_sector(mf,i=0,j=0)
        print("U=%4.1f %-14s conv=%s  |Delta|=%.8f  ee diag=%s  Etot=%.8f"%(
            U,nm,s.converged,np.max(np.abs(eh)),np.round(np.diag(ee).real,6),s.total_energy))
    print()
```

## `p10_keldysh.py`

Backs: **L3 transport dagger** — CLEARED: Floquet-Keldysh dc_current with a genuinely complex non-Hermitian Nambu lead coupling

```python
import numpy as np,sys
from scipy.integrate import quad
from pyqula import geometry,heterostructures,algebra
from pyqula.operators import get_electron,get_hole

def tauz(h): return np.array((get_electron(h)-get_hole(h)).todense())

def static_ref(h1,h2,transp,V,delta,central=None):
    tz=tauz(h1)
    a=h1.copy(); a.intra=a.intra+(V/2)*tz
    b=h2.copy(); b.intra=b.intra-(V/2)*tz
    kw={}
    if central is not None:
        kw["central"]=[ (lambda hc: (hc.copy(), None))(hc)[0] for hc in central]
        sh=[]
        for hc in central:
            c=hc.copy(); c.intra=c.intra+(V/2)*tz; sh.append(c)
        kw["central"]=sh
    HTb=heterostructures.build(a,b,**kw); HTb.set_coupling(transp); HTb.delta=delta
    val,_=quad(lambda e: HTb.didv(energy=e),-abs(V)/2,abs(V)/2,limit=100,epsrel=1e-5)
    return val*np.sign(V)

def lead():
    g=geometry.chain(2)
    h=g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.35)
    h.add_zeeman([0.12,0.08,0.05])
    h.turn_nambu()
    return h

h1=lead(); h2=h1.copy()
HT=heterostructures.build(h1.copy(),h2.copy())
rc=np.array(algebra.todense(HT.right_coupling))
print("coupling shape",rc.shape)
print("||rc-rc^dag||",np.max(np.abs(rc-rc.conj().T)))
print("||rc-rc^T||  ",np.max(np.abs(rc-rc.T)))
print("||rc-rc^*||  ",np.max(np.abs(rc-rc.conj())))
for transp in [0.4,1.0]:
  for V in [0.3,-0.3]:
    HT=heterostructures.build(h1.copy(),h2.copy())
    HT.set_coupling(transp); HT.delta=1e-4
    I=HT.get_dc_current(V,nmax=6,nmax_max=24,tol=1e-4)
    Ir=static_ref(h1,h2,transp,V,1e-4)
    print("NO-CENTRAL  transp=%.1f V=%+.1f   I=%.6f  Iref=%.6f  rel=%.2e"%(transp,V,I,Ir,abs(I-Ir)/max(abs(Ir),1e-12)))
```

## `p11_keldysh_cplx.py`

Backs: **L3 transport dagger** — CLEARED: Floquet-Keldysh dc_current with a genuinely complex non-Hermitian Nambu lead coupling

```python
import numpy as np
from scipy.integrate import quad
from pyqula import geometry,heterostructures,algebra
from pyqula.operators import get_electron,get_hole
from pyqula.multihopping import MultiHopping

def tauz(h): return np.array((get_electron(h)-get_hole(h)).todense())
def static_ref(h1,h2,transp,V,delta,central=None):
    tz=tauz(h1)
    a=h1.copy(); a.intra=a.intra+(V/2)*tz
    b=h2.copy(); b.intra=b.intra-(V/2)*tz
    kw={}
    if central is not None:
        sh=[]
        for hc in central:
            c=hc.copy(); c.intra=c.intra+(V/2)*tz; sh.append(c)
        kw["central"]=sh
    HTb=heterostructures.build(a,b,**kw); HTb.set_coupling(transp); HTb.delta=delta
    val,_=quad(lambda e: HTb.didv(energy=e),-abs(V)/2,abs(V)/2,limit=100,epsrel=1e-5)
    return val*np.sign(V)

def rand_unitary(n,seed):
    rs=np.random.RandomState(seed)
    A=rs.randn(n,n)+1j*rs.randn(n,n)
    q,r=np.linalg.qr(A); return q@np.diag(np.diag(r)/np.abs(np.diag(r)))

def lead(rot=False,seed=13):
    g=geometry.chain(2)
    h=g.get_hamiltonian(has_spin=True)
    h.add_rashba(0.35); h.add_zeeman([0.12,0.08,0.05])
    if rot:
        n=h.intra.shape[0]; U=rand_unitary(n,seed)
        mh=h.get_multihopping().get_dict()
        d={k:U.conj().T@np.array(algebra.todense(v))@U for k,v in mh.items()}
        h.set_multihopping(MultiHopping(d))
        h=h.get_no_multicell()
    h.turn_nambu()
    return h

h1=lead(rot=True); h2=h1.copy()
HT=heterostructures.build(h1.copy(),h2.copy())
rc=np.array(algebra.todense(HT.right_coupling))
print("||rc-rc^dag||=%.3f  ||rc-rc^T||=%.3f  ||rc-rc^*||=%.3f"%(
  np.max(np.abs(rc-rc.conj().T)),np.max(np.abs(rc-rc.T)),np.max(np.abs(rc-rc.conj()))))
hc=h1.copy(); hc.intra=hc.intra+0.5*tauz(h1)
for tag,central in [("NO-CENTRAL",None),("1-CENTRAL",[hc]),("2-CENTRAL",[hc,hc])]:
  for transp in [0.5,1.0]:
    V=0.3
    kw={} if central is None else {"central":[c.copy() for c in central]}
    HT=heterostructures.build(h1.copy(),h2.copy(),**kw)
    HT.set_coupling(transp); HT.delta=1e-4
    I=HT.get_dc_current(V,nmax=6,nmax_max=24,tol=1e-4)
    Ir=static_ref(h1,h2,transp,V,1e-4,central=central)
    print("%s transp=%.1f  I=%.6f Iref=%.6f rel=%.2e"%(tag,transp,I,Ir,abs(I-Ir)/max(abs(Ir),1e-12)))
```

## `p12_multiterminal.py`

Backs: **L3 transport dagger** — multiterminal.Device.transmission is non-functional: neighbor.parametric_hopping cannot build a rectangular lead-to-central coupling

```python
import numpy as np
from pyqula import geometry, multiterminal
g = geometry.honeycomb_lattice()
gc = g.supercell(3); gc.dimensionality=0
gl = g.supercell(2)
d = multiterminal.Device()
d.biterminal(right_g=gl,left_g=gl,central_g=gc)
print("built device; intra shape",d.intra.shape)
try:
    print("transmission:",d.transmission(energy=0.1))
except Exception as e:
    print("RAISED:",type(e).__name__,":",e)
l=d.leads[0]
print("lead.intra",np.shape(l.intra),"lead.inter",np.shape(l.inter),"coupling",np.shape(l.coupling))
gr=l.get_green(0.1)
print("gr",np.shape(gr))
print("correct order t@gr@t^dag ->",np.shape(np.asarray(l.coupling)@np.asarray(gr)@np.asarray(l.coupling).conj().T))
# now patch to the correct order and see what breaks next
import pyqula.multiterminal as mt
def patched(self,energy,error=1e-4,delta=1e-4):
    gr=self.get_green(energy,error=error,delta=delta)
    t=np.asarray(self.coupling)
    return t@np.asarray(gr)@t.conj().T
mt.Lead.get_selfenergy=patched
try:
    print("transmission after fixing the coupling order:",d.transmission(energy=0.1))
except Exception as e:
    print("STILL RAISES:",type(e).__name__,":",e)
```

## `p13_unitarity.py`

Backs: **L3 transport dagger** — CLEARED: Landauer / get_smatrix / didv on a fully complex non-Hermitian 2-orbital junction

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula import heterostructures
from pyqula.transporttk.smatrix import get_smatrix
TA=make_T(seed=1);HA=make_H0(seed=2)
TC=make_T(seed=11);HC=make_H0(seed=12)
hA=build_h(TA,HA); hC=build_h(TC,HC)
def report(ht,tag):
    print("---",tag,"block_diagonal=",ht.block_diagonal)
    for E in [-2.5,-1.9,-1.0,1.8,2.7]:
        s=get_smatrix(ht,energy=E,check=False)
        S=np.block([[np.array(s[0][0]),np.array(s[0][1])],[np.array(s[1][0]),np.array(s[1][1])]])
        u=np.max(np.abs(S@S.conj().T-np.identity(S.shape[0])))
        tLR=np.trace(np.array(s[1][0])@np.array(s[1][0]).conj().T).real
        tRL=np.trace(np.array(s[0][1])@np.array(s[0][1]).conj().T).real
        L=ht.landauer(energy=E)
        print("  E=%+.2f  |SS^dag-1|=%.2e  T_LR=%.6f T_RL=%.6f  landauer=%.6f"%(E,u,tLR,tRL,L))
for nc in [1,2,3]:
    ht=heterostructures.build(left=hA,right=hA,central=[hC]*nc); ht.delta=1e-7
    report(ht,"central x%d"%nc)
```

## `p14_aaa.py`

Backs: **L3 transport dagger** — CLEARED: Batched Sancho-Rubio self-energy and the AAA interpolant's agreement with direct solves

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula import heterostructures
from pyqula.aaatk.selfenergy_aaa import SelfenergyAAA
TA=make_T(seed=1);HA=make_H0(seed=2)
hA=build_h(TA,HA)
ht=heterostructures.build(left=hA,right=hA,central=[hA]); ht.delta=1e-4
delta=ht.delta
for lead in [0,1]:
    def get_se(e,lead=lead): return ht.get_selfenergy(e,lead=lead,delta=delta,pristine=True,numba=True)
    def get_se_b(es,lead=lead): return ht.get_selfenergy_batch(es,lead=lead,delta=delta,pristine=True)
    dim=np.array(get_se(0.)).shape[0]
    S=SelfenergyAAA(get_se,dim,-3.5,3.5,delta,tolerance=1e-6,get_selfenergy_batch=get_se_b)
    errs=[]
    for e in np.linspace(-3.4,3.4,41):
        a=np.array(S(e)); b=np.array(get_se(e))
        errs.append(np.max(np.abs(a-b)))
    print("lead",lead,"max|AAA-direct| over window:",max(errs))
    # batched vs direct selfenergy
    es=np.linspace(-3.4,3.4,17)
    bb=np.array(ht.get_selfenergy_batch(es,lead=lead,delta=delta,pristine=True))
    dd=np.array([np.array(get_se(e)) for e in es])
    print("     max|batch-direct|:",np.max(np.abs(bb-dd)))
```

## `p15_central.py`

Backs: **L3 transport dagger** — CLEARED: transporttk/central.py get_central_heterostructure with a non-Hermitian lead coupling

```python
import numpy as np
from pyqula import geometry,algebra
from pyqula.transporttk.central import get_central_heterostructure
def lead(lam=0.4):
    h=geometry.chain().get_hamiltonian(has_spin=True)
    h.add_rashba(lam)
    return h
h=lead()
inter=np.array(algebra.todense(h.get_no_multicell().inter))
print("lead inter:\n",inter)
print("||inter-inter^dag||",np.max(np.abs(inter-inter.conj().T)))
# finite chain of the same material
N=6
g=geometry.chain().get_supercell(N); g.dimensionality=0
hc=g.get_hamiltonian(has_spin=True)
hc.add_rashba(0.4)
ht=get_central_heterostructure(hc,i=0,j=N-1,left=lead(),right=lead())
ht.delta=1e-6
def nch(E):
    # 2-band spinful Rashba chain: count right movers
    hk=h.get_hk_gen()
    ks=np.linspace(0,1,4001)
    es=np.array([np.linalg.eigvalsh(np.array(hk([k,0,0]))) for k in ks])
    c=0
    for ib in range(es.shape[1]):
        b=es[:,ib]; c+=np.sum((b[:-1]-E)*(b[1:]-E)<0)
    return c//2
for E in [-1.5,-0.8,0.0,0.5,1.4,2.1]:
    T=float(np.real(np.sum(ht.landauer(E))))
    print("E=%+.2f  T=%.6f  nchan=%d"%(E,T,nch(E)))
```

## `p16_fulladaptive.py`

Backs: **L3 transport dagger** — bloch_selfenergy(mode="full_adaptive") in 2D hardcodes eps=0.1 and silently ignores the caller's `error`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0
from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.green import bloch_selfenergy
n=2
H0=make_H0(seed=2); Tx=make_T(seed=1); Ty=0.7*make_T(seed=4); Txy=0.35*make_T(seed=6)
g=geometry.square_lattice()
h=g.get_hamiltonian(has_spin=True)
d={(0,0,0):np.array(H0,dtype=np.complex128),(1,0,0):np.array(Tx),(-1,0,0):np.array(Tx).conj().T,
   (0,1,0):np.array(Ty),(0,-1,0):np.array(Ty).conj().T,(1,1,0):np.array(Txy),(-1,-1,0):np.array(Txy).conj().T}
h.set_multihopping(MultiHopping(d))
E,delta=0.17,0.08
hk=h.get_hk_gen(); nk=600
acc=np.zeros((n,n),dtype=np.complex128)
for a in range(nk):
  for b in range(nk):
    acc+=np.linalg.inv((E+1j*delta)*np.identity(n)-np.array(hk([a/nk,b/nk,0.])))
acc/=nk*nk
for err in [1e-2,1e-4,1e-8]:
    g1=np.array(bloch_selfenergy(h,energy=E,delta=delta,mode="full_adaptive",error=err)[0])
    print("full_adaptive error=%g -> maxerr vs brute = %.4e"%(err,np.max(np.abs(g1-acc))))
for err in [1e-2,1e-6]:
    g1=np.array(bloch_selfenergy(h,energy=E,delta=delta,mode="adaptive",error=err)[0])
    print("adaptive      error=%g -> maxerr vs brute = %.4e"%(err,np.max(np.abs(g1-acc))))
```

## `p17_rg_batch.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0
from pyqula.greentk.rg import (green_renormalization,green_renormalization_jit_batch,
    surface_dyson_residual,green_renormalization_python,green_renormalization_jit)
n=2
T=make_T(seed=1)
def finite(H0,T,N,E,delta):
    M=np.zeros((N*n,N*n),dtype=np.complex128)
    for i in range(N): M[i*n:(i+1)*n,i*n:(i+1)*n]=H0
    for i in range(N-1):
        M[i*n:(i+1)*n,(i+1)*n:(i+2)*n]=T
        M[(i+1)*n:(i+2)*n,i*n:(i+1)*n]=T.conj().T
    G=np.linalg.inv((E+1j*delta)*np.identity(N*n)-M)
    return G[0:n,0:n],G[(N//2)*n:(N//2+1)*n,(N//2)*n:(N//2+1)*n]
print("=== pathological: H0=0, E=0, tiny delta, non-hermitian T ===")
H0=np.zeros((n,n),dtype=np.complex128)
for delta in [1e-3,1e-6,1e-10]:
    gs_ref,gb_ref=finite(H0,T,4000,0.0,delta)
    gb,gs=green_renormalization(H0,T,energy=0.0,delta=delta)
    gbj,gsj=green_renormalization(H0,T,energy=0.0,delta=delta,numba=True)
    print(" delta=%g  |gs-ref|=%.3e  |gb-ref|=%.3e  (jit: %.3e, %.3e)  resid=%.2e"%(
      delta,np.max(np.abs(gs-gs_ref)),np.max(np.abs(gb-gb_ref)),
      np.max(np.abs(gsj-gs_ref)),np.max(np.abs(gbj-gb_ref)),
      surface_dyson_residual(gs,H0,T,(0.0+1j*delta)*np.identity(n))))
print("=== batch vs scalar, generic H0, non-hermitian T ===")
H0=make_H0(seed=2)
es=np.array([-2.1,-1.0,0.0,0.3,1.7,2.9])
gbb,gsb=green_renormalization_jit_batch(H0,T,es,delta=1e-4)
for i,e in enumerate(es):
    gb,gs=green_renormalization(H0,T,energy=e,delta=1e-4)
    print("  E=%+.2f  |gs_batch-gs|=%.3e |gb_batch-gb|=%.3e"%(e,np.max(np.abs(gsb[i]-gs)),np.max(np.abs(gbb[i]-gb))))
```

## `p17b.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0
from pyqula.greentk.rg import (green_renormalization,green_renormalization_jit_batch,
    surface_dyson_residual,surface_green_dyson)
n=2;T=make_T(seed=1)
print("=== pathological E=0, H0=0, non-hermitian T ===",flush=True)
H0=np.zeros((n,n),dtype=np.complex128)
for delta in [1e-3,1e-6,1e-10,1e-12]:
    ez=(0.0+1j*delta)*np.identity(n)
    gb,gs=green_renormalization(H0,T,energy=0.0,delta=delta)
    gfp=surface_green_dyson(H0,T,ez)
    print(" delta=%-8g resid(gs)=%.2e resid(fp)=%.2e |gs-fp|=%.2e"%(
      delta,surface_dyson_residual(gs,H0,T,ez),surface_dyson_residual(gfp,H0,T,ez),
      np.max(np.abs(gs-gfp))),flush=True)
print("=== batch vs scalar ===",flush=True)
H0=make_H0(seed=2)
es=np.array([-2.1,-1.0,0.0,0.3,1.7,2.9])
gbb,gsb=green_renormalization_jit_batch(H0,T,es,delta=1e-4)
for i,e in enumerate(es):
    gb,gs=green_renormalization(H0,T,energy=e,delta=1e-4)
    gbj,gsj=green_renormalization(H0,T,energy=e,delta=1e-4,numba=True)
    print("  E=%+.2f |gs_b-gs|=%.2e |gb_b-gb|=%.2e |gs_jit-gs|=%.2e"%(
      e,np.max(np.abs(gsb[i]-gs)),np.max(np.abs(gbb[i]-gb)),np.max(np.abs(gsj-gs))),flush=True)
print("=== finite-chain oracle, moderate delta ===",flush=True)
def finite(H0,T,N,E,delta):
    M=np.zeros((N*n,N*n),dtype=np.complex128)
    for i in range(N): M[i*n:(i+1)*n,i*n:(i+1)*n]=H0
    for i in range(N-1):
        M[i*n:(i+1)*n,(i+1)*n:(i+2)*n]=T
        M[(i+1)*n:(i+2)*n,i*n:(i+1)*n]=T.conj().T
    G=np.linalg.inv((E+1j*delta)*np.identity(N*n)-M)
    return G[0:n,0:n],G[(N//2)*n:(N//2+1)*n,(N//2)*n:(N//2+1)*n]
for E in [0.0,1.3]:
    gsr,gbr=finite(H0,T,500,E,0.05)
    gb,gs=green_renormalization(H0,T,energy=E,delta=0.05)
    print("  E=%.2f |gs-ref|=%.2e |gb-ref|=%.2e"%(E,np.max(np.abs(gs-gsr)),np.max(np.abs(gb-gbr))),flush=True)
```

## `p18_dyson_hkgen.py`

Backs: **L3 transport dagger** — dyson1d_hkgen calls the 11-argument jitted kernel with 7 arguments — hard TypeError on a live Embedding path

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0
from pyqula import geometry,embedding
from pyqula.multihopping import MultiHopping
n=2
H0=make_H0(seed=2);T1=make_T(seed=1);T2=0.5*make_T(seed=5)
g=geometry.chain().get_supercell(2)
h=g.get_hamiltonian(has_spin=False)
d={(0,0,0):np.array(H0,dtype=np.complex128),
   (1,0,0):np.array(T1),(-1,0,0):np.array(T1).conj().T,
   (2,0,0):np.array(T2),(-2,0,0):np.array(T2).conj().T}
h.set_multihopping(MultiHopping(d))
print("is_multicell",h.is_multicell)
from pyqula.htk.kchain import detect_longest_hopping
print("longest hopping",detect_longest_hopping(h))
try:
    hnm=h.get_no_multicell(); print("get_no_multicell OK -> is_multicell",hnm.is_multicell)
except Exception as e:
    print("get_no_multicell RAISED:",type(e).__name__,e)
from pyqula.dyson import dyson
try:
    out=dyson(h,[2,1],200,0.2+0.05j); print("dyson nsuper=2 OK, shape",np.shape(out))
except Exception as e:
    print("dyson RAISED:",type(e).__name__,":",e)
eo=embedding.Embedding(h)
try:
    gg=eo.get_gf(energy=0.2,delta=0.05,nsuper=2,nk=200); print("Embedding.get_gf nsuper=2 OK",np.shape(gg))
except Exception as e:
    print("Embedding.get_gf RAISED:",type(e).__name__,":",e)
```

## `p19_degenerate.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T
from pyqula.greentk.rg import green_renormalization,surface_dyson_residual,surface_green_dyson
import mpmath as mp
mp.mp.dps=60

def mp_decimation(H0,T,E,delta,nite=200):
    n=H0.shape[0]
    def M(a): return mp.matrix([[mp.mpc(a[i,j]) for j in range(a.shape[1])] for i in range(a.shape[0])])
    I=mp.eye(n)
    e=I*mp.mpc(E,delta)
    alpha=M(T); beta=M(T.conj().T); eps=M(H0); epss=M(H0)
    for _ in range(nite):
        g=mp.inverse(e-eps)
        ag=alpha*g; bg=beta*g
        epss=epss+ag*beta
        eps=eps+ag*beta+bg*alpha
        al=ag*alpha; be=bg*beta
        alpha=al; beta=be
        if max(abs(alpha[i,j]) for i in range(n) for j in range(n))<mp.mpf(10)**(-40): break
    gs=mp.inverse(e-epss)
    return np.array([[complex(gs[i,j]) for j in range(n)] for i in range(n)])

for tag,(H0,T) in [("1-orbital chain",(np.zeros((1,1),dtype=complex),np.array([[1.0+0j]]))),
                   ("2-orb non-herm  ",(np.zeros((2,2),dtype=complex),make_T(seed=1)))]:
    n=H0.shape[0]
    print("=====",tag)
    for delta in [1e-3,1e-5,1e-6,1e-8,1e-10,1e-12]:
        ez=(0.0+1j*delta)*np.identity(n)
        ref=mp_decimation(H0,T,0.0,delta)
        try:
            gb,gs=green_renormalization(H0,T,energy=0.0,delta=delta)
            err=np.max(np.abs(gs-ref))
            print("  delta=%-7g  |g_pyqula - g_exact|=%.3e   |g_exact|=%.3f  resid=%.2e"%(
              delta,err,np.max(np.abs(ref)),surface_dyson_residual(gs,H0,T,ez)))
        except Exception as ex:
            print("  delta=%-7g  RAISED %s: %s   (|g_exact|=%.3f)"%(delta,type(ex).__name__,ex,np.max(np.abs(ref))))
```

## `p20_ssh.py`

Backs: **L3 transport dagger** — CLEARED: The SSH chain — the obvious physical case for the degenerate-energy decimation failure — does NOT crash

```python
import numpy as np
from pyqula import geometry,heterostructures,algebra
from pyqula.multihopping import MultiHopping
from pyqula.greentk.rg import green_renormalization
def ssh(v=0.4,w=1.0):
    g=geometry.chain().get_supercell(2)
    h=g.get_hamiltonian(has_spin=False)
    d={(0,0,0):np.array([[0,v],[v,0]],dtype=complex),
       (1,0,0):np.array([[0,0],[w,0]],dtype=complex),
       (-1,0,0):np.array([[0,w],[0,0]],dtype=complex)}
    h.set_multihopping(MultiHopping(d)); return h
h=ssh()
hnm=h.get_no_multicell()
H0=np.array(algebra.todense(hnm.intra)); T=np.array(algebra.todense(hnm.inter))
print("H0=\n",H0,"\nT=\n",T)
for delta in [1e-4,1e-6,1e-8,1e-12]:
    try:
        gb,gs=green_renormalization(H0,T,energy=0.0,delta=delta)
        print(" raw rg delta=%-8g ok, max|gs|=%.3e"%(delta,np.max(np.abs(gs))))
    except Exception as e:
        print(" raw rg delta=%-8g RAISED %s: %s"%(delta,type(e).__name__,e))
ht=heterostructures.build(left=h,right=h,central=[h])
ht.delta=1e-6
print("--- user-facing calls at E=0 (smatrix forces delta=1e-12) ---")
for name,f in [("landauer",lambda: ht.landauer(energy=0.0)),
               ("didv",lambda: ht.didv(energy=0.0)),
               ("get_smatrix",lambda: __import__("pyqula.transporttk.smatrix",fromlist=["x"]).get_smatrix(ht,energy=0.0))]:
    try: print("  %s(0.0) = %s"%(name,f()))
    except Exception as e: print("  %s(0.0) RAISED %s: %s"%(name,type(e).__name__,e))
```

## `p21_userfacing_crash.py`

Backs: **L3 transport dagger** — didv/get_smatrix crash with LinAlgError on a 2-orbital lead whose surface Green's function diverges at the evaluated energy

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,build_h
from pyqula import heterostructures,embedding
T=make_T(seed=1); H0=np.zeros((2,2),dtype=complex)
h=build_h(T,H0)
hk=h.get_hk_gen()
print("H(k) hermitian at k=0.3:",np.max(np.abs(np.array(hk([0.3,0,0]))-np.array(hk([0.3,0,0])).conj().T)))
ht=heterostructures.build(left=h,right=h,central=[h]); ht.delta=1e-6
for name,f in [("landauer(E=0)",lambda: ht.landauer(energy=0.0)),
               ("didv(E=0)    ",lambda: ht.didv(energy=0.0)),
               ("didv(E=0.5)  ",lambda: ht.didv(energy=0.5)),
               ("Embedding.get_gf(E=0,delta=1e-9)",
                lambda: embedding.Embedding(h).get_gf(energy=0.0,delta=1e-9,nk=100))]:
    try: print("  %s -> %s"%(name,np.round(np.array(f()).real,8) if not np.isscalar(f()) else f()))
    except Exception as e: print("  %s RAISED %s: %s"%(name,type(e).__name__,e))
```

## `p21b_traceback.py`

```python
import numpy as np,sys,traceback
sys.path.insert(0,"SCRATCH")
from fixture import make_T,build_h
from pyqula import heterostructures
T=make_T(seed=1); H0=np.zeros((2,2),dtype=complex)
h=build_h(T,H0)
ht=heterostructures.build(left=h,right=h,central=[h]); ht.delta=1e-6
ht.didv(energy=0.0)
```

## `p22_2d.py`

Backs: **L3 transport dagger** — CLEARED: 2D (quasi-1D) heterostructures.build at fixed transverse k

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0
from pyqula import geometry,heterostructures,algebra
from pyqula.multihopping import MultiHopping
n=2
H0=make_H0(seed=2);Tx=make_T(seed=1);Ty=0.7*make_T(seed=4);Txy=0.35*make_T(seed=6)
def mk(conj=False):
    g=geometry.square_lattice()
    h=g.get_hamiltonian(has_spin=True)
    d={(0,0,0):np.array(H0,dtype=np.complex128),(1,0,0):np.array(Tx),(-1,0,0):np.array(Tx).conj().T,
       (0,1,0):np.array(Ty),(0,-1,0):np.array(Ty).conj().T,(1,1,0):np.array(Txy),(-1,-1,0):np.array(Txy).conj().T}
    if conj: d={( -k[0],-k[1],-k[2]):v for k,v in d.items()}
    h.set_multihopping(MultiHopping(d)); return h
h2=mk()
ht=heterostructures.build(left=h2,right=h2,central=[h2])
htm=heterostructures.build(left=mk(True),right=mk(True),central=[mk(True)])
def nch(h1d,E):
    hk=h1d.get_hk_gen()
    ks=np.linspace(0,1,6001)
    es=np.array([np.linalg.eigvalsh(np.array(hk([k,0,0]))) for k in ks])
    c=0
    for ib in range(es.shape[1]):
        b=es[:,ib]; c+=np.sum((b[:-1]-E)*(b[1:]-E)<0)
    return c//2
print("2D junction: landauer at fixed k vs open-channel count of get_1dh(k)")
for k in [0.0,0.17,0.33,0.5,0.71]:
    h1d=h2.get_1dh(k)
    hti=ht.generate([k,1.]); hti.delta=1e-5
    htmi=htm.generate([k,1.]); htmi.delta=1e-5
    for E in [-2.0,0.0,2.0]:
        T=float(np.real(hti.landauer(energy=E)))
        Tm=float(np.real(htmi.landauer(energy=E)))
        nc=nch(h1d,E)
        ok="" if abs(T-nc)<3e-3 else "   <-- MISMATCH"
        print("  k=%.2f E=%+.1f  T=%.6f  nchan=%d   T_mirror=%.6f%s"%(k,E,T,nc,Tm,ok))
```

## `p23_scan.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0,build_h
from pyqula import heterostructures
for seed,tag,H0 in [(1,"H0=0      ",np.zeros((2,2),dtype=complex)),
                    (1,"H0 generic",make_H0(seed=2)),
                    (7,"H0=0 s7   ",np.zeros((2,2),dtype=complex)),
                    (7,"H0 gen s7 ",make_H0(seed=8))]:
    T=make_T(seed=seed); h=build_h(T,H0)
    ht=heterostructures.build(left=h,right=h,central=[h]); ht.delta=1e-6
    bad=[]
    for E in np.linspace(-3.,3.,61):
        try: ht.didv(energy=float(E))
        except Exception as e: bad.append((round(float(E),3),type(e).__name__))
    print("%s seed=%d  crashes at %d of 61 energies: %s"%(tag,seed,len(bad),bad[:8]))
```

## `p24_aaa_conv.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0,build_h
from pyqula import heterostructures
from pyqula.aaatk.selfenergy_aaa import SelfenergyAAA
h=build_h(make_T(seed=1),make_H0(seed=2))
ht=heterostructures.build(left=h,right=h,central=[h]); ht.delta=1e-4
d=ht.delta
def gse(e): return ht.get_selfenergy(e,lead=0,delta=d,pristine=True,numba=True)
def gseb(es): return ht.get_selfenergy_batch(es,lead=0,delta=d,pristine=True)
for tol in [1e-3,1e-6]:
    S=SelfenergyAAA(gse,2,-3.5,3.5,d,tolerance=tol,get_selfenergy_batch=gseb)
    es=np.linspace(-3.4,3.4,41)
    ref=np.array([np.array(gse(e)) for e in es])
    got=np.array([np.array(S(e)) for e in es])
    scale=np.max(np.abs(ref))
    print("tol=%g converged=%s  max_abs=%.3e  max_rel(/max|Sigma|=%.3f)=%.3e"%(
        tol,getattr(S,"converged","?"),np.max(np.abs(got-ref)),scale,np.max(np.abs(got-ref))/scale))
```

## `p25_aaa_1orb.py`

Backs: **L3 transport dagger** — CLEARED: Batched Sancho-Rubio self-energy and the AAA interpolant's agreement with direct solves

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0,build_h
from pyqula import geometry,heterostructures
from pyqula.aaatk.selfenergy_aaa import SelfenergyAAA
def check(h,dim,tag):
    ht=heterostructures.build(left=h,right=h,central=[h]); ht.delta=1e-4
    d=ht.delta
    gse=lambda e: ht.get_selfenergy(e,lead=0,delta=d,pristine=True,numba=True)
    gseb=lambda es: ht.get_selfenergy_batch(es,lead=0,delta=d,pristine=True)
    for tol in [1e-3,1e-6]:
        S=SelfenergyAAA(gse,dim,-3.5,3.5,d,tolerance=tol,get_selfenergy_batch=gseb)
        es=np.linspace(-3.4,3.4,401)
        ref=np.array([np.array(gse(e)) for e in es]); got=np.array([np.array(S(e)) for e in es])
        sc=np.max(np.abs(ref))
        print("%s tol=%g conv=%s  max_rel=%.3e  (miss factor %.1f)"%(tag,tol,S.converged,
             np.max(np.abs(got-ref))/sc,np.max(np.abs(got-ref))/sc/tol))
check(geometry.chain().get_hamiltonian(has_spin=False),1,"1-orbital chain ")
check(build_h(make_T(seed=1),make_H0(seed=2)),2,"2-orbital nonherm")
```

## `p26_aaa_det.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0,build_h
from pyqula import heterostructures
from pyqula.aaatk.selfenergy_aaa import SelfenergyAAA
h=build_h(make_T(seed=1),make_H0(seed=2))
ht=heterostructures.build(left=h,right=h,central=[h]); ht.delta=1e-4
d=ht.delta
gse=lambda e: ht.get_selfenergy(e,lead=0,delta=d,pristine=True,numba=True)
gseb=lambda es: ht.get_selfenergy_batch(es,lead=0,delta=d,pristine=True)
for rep in range(2):
  for tol in [1e-3,1e-6]:
    S=SelfenergyAAA(gse,2,-3.5,3.5,d,tolerance=tol,get_selfenergy_batch=gseb)
    out=[]
    for N in [41,401,4001]:
        es=np.linspace(-3.4,3.4,N)
        ref=np.array([np.array(gse(e)) for e in es]); got=np.array([np.array(S(e)) for e in es])
        out.append(np.max(np.abs(got-ref))/np.max(np.abs(ref)))
    print("rep=%d tol=%g conv=%s nsolved=%d  max_rel[41,401,4001]=%s"%(
        rep,tol,S.converged,len(S._solved),["%.2e"%x for x in out]))
```

## `p27_aaa_spike.py`

Backs: **L3 transport dagger** — SelfenergyAAA reports converged=True while the local relative error is 7.7% — its tolerance is normalized by the window-maximum |Sigma|

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0,build_h
from pyqula import geometry,heterostructures
from pyqula.aaatk.selfenergy_aaa import SelfenergyAAA
def run(h,dim,tag):
    ht=heterostructures.build(left=h,right=h,central=[h]); ht.delta=1e-4
    d=ht.delta
    gse=lambda e: ht.get_selfenergy(e,lead=0,delta=d,pristine=True,numba=True)
    gseb=lambda es: ht.get_selfenergy_batch(es,lead=0,delta=d,pristine=True)
    S=SelfenergyAAA(gse,dim,-3.5,3.5,d,tolerance=1e-3,get_selfenergy_batch=gseb)
    es=np.linspace(-3.4,3.4,41)
    errs=[np.max(np.abs(np.array(S(e))-np.array(gse(e)))) for e in es]
    i=int(np.argmax(errs)); e0=es[i]
    print("%s worst 41-grid point E=%.6f  abs err=%.3e (|Sigma|~%.3f)"%(tag,e0,errs[i],np.max(np.abs(np.array(gse(e0))))))
    for dd in [0.,1e-12,1e-10,1e-8,1e-6,1e-4,1e-3]:
        e=e0+dd
        print("     E=E0+%-8g err=%.3e"%(dd,np.max(np.abs(np.array(S(e))-np.array(gse(e))))))
run(geometry.chain().get_hamiltonian(has_spin=False),1,"1-orb  ")
run(build_h(make_T(seed=1),make_H0(seed=2)),2,"2-orbNH")
```

## `p3_surface.py`

Backs: **L3 transport dagger** — CLEARED: green_renormalization's left-vs-right surface convention

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula.greentk.rg import green_renormalization,surface_dyson_residual
T=make_T();H0=make_H0();n=2
N=400;delta=0.05;E=0.13
M=np.zeros((N*n,N*n),dtype=np.complex128)
for i in range(N): M[i*n:(i+1)*n,i*n:(i+1)*n]=H0
for i in range(N-1):
    M[i*n:(i+1)*n,(i+1)*n:(i+2)*n]=T
    M[(i+1)*n:(i+2)*n,i*n:(i+1)*n]=T.conj().T
G=np.linalg.inv((E+1j*delta)*np.identity(N*n)-M)
g00=G[0:n,0:n]; gNN=G[(N-1)*n:,(N-1)*n:]
gb_T,gs_T=green_renormalization(H0,T,energy=E,delta=delta)
gb_Td,gs_Td=green_renormalization(H0,T.conj().T,energy=E,delta=delta)
f=lambda a,b: np.max(np.abs(a-b))
print("finite g00  vs gr(H0,T).surf   :",f(g00,gs_T))
print("finite g00  vs gr(H0,Td).surf  :",f(g00,gs_Td))
print("finite gNN  vs gr(H0,T).surf   :",f(gNN,gs_T))
print("finite gNN  vs gr(H0,Td).surf  :",f(gNN,gs_Td))
mid=N//2
gmid=G[mid*n:(mid+1)*n,mid*n:(mid+1)*n]
print("finite gmid vs gr(H0,T).bulk   :",f(gmid,gb_T))
print("finite gmid vs gr(H0,Td).bulk  :",f(gmid,gb_Td))
print("residual(T) :",surface_dyson_residual(gs_T,H0,T,(E+1j*delta)*np.identity(n)))
print("residual(Td):",surface_dyson_residual(gs_Td,H0,T.conj().T,(E+1j*delta)*np.identity(n)))
```

## `p4_landauer.py`

Backs: **L3 transport dagger** — CLEARED: Landauer / get_smatrix / didv on a fully complex non-Hermitian 2-orbital junction

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula import heterostructures
T=make_T();H0=make_H0();n=2
h=build_h(T,H0)
# channel count from the dispersion: number of k with band energy == E
def nchannels(E,nk=200001):
    ks=np.linspace(0,1,nk)
    es=np.array([np.linalg.eigvalsh(H0+T*np.exp(1j*2*np.pi*k)+T.conj().T*np.exp(-1j*2*np.pi*k)) for k in np.linspace(0,1,4001)])
    c=0
    for ib in range(n):
        b=es[:,ib]
        c+=np.sum((b[:-1]-E)*(b[1:]-E)<0)
    return c//2   # crossings come in pairs (+k,-k); right movers = half
ht=heterostructures.build(left=h,right=h,central=[h,h,h])
ht.delta=1e-5
print("intercell T non-hermiticity:",np.max(np.abs(T-T.conj().T)))
bad=0
for E in np.linspace(-3.2,3.2,33):
    G=ht.landauer(energy=E)
    nc=nchannels(E)
    flag="" 
    if abs(G-nc)>1e-3: flag="  <-- MISMATCH"; bad+=1
    print("E=%+.3f  T=%.6f   nchan=%d%s"%(E,G,nc,flag))
print("mismatches:",bad)
```

## `p5_smatrix_mirror.py`

Backs: **L3 transport dagger** — CLEARED: Landauer / get_smatrix / didv on a fully complex non-Hermitian 2-orbital junction

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula import heterostructures
from pyqula.multihopping import MultiHopping

def mirror(h):
    """reverse the chain: T -> T^dag"""
    mh=h.get_multihopping().get_dict()
    d={}
    for k,v in mh.items():
        d[(-k[0],-k[1],-k[2])]=np.array(v)
    h2=h.copy(); h2.set_multihopping(MultiHopping(d)); return h2

TA=make_T(seed=1);HA=make_H0(seed=2)
TB=0.8*make_T(seed=7);HB=make_H0(seed=8)
TC=make_T(seed=11);HC=make_H0(seed=12)
hA=build_h(TA,HA); hB=build_h(TB,HB); hC=build_h(TC,HC)

def Tsm(ht,E):
    from pyqula.heterostructures import get_smatrix
    s=get_smatrix(ht,energy=E)
    t=s[1][0]
    return np.trace(t@t.conj().T).real
print("=== smatrix vs landauer, symmetric junction, non-hermitian T ===")
ht=heterostructures.build(left=hA,right=hA,central=[hC])
ht.delta=1e-6
for E in [-2.5,-1.9,-1.0,0.6,1.8,2.7,3.0]:
    L=ht.landauer(energy=E); S=Tsm(ht,E)
    print("E=%+.2f  landauer=%.6f  smatrix=%.6f  diff=%.2e"%(E,L,S,abs(L-S)))
print()
print("=== mirror invariance:  build(A,B,[C])  vs  build(mB,mA,[mC]) ===")
ht1=heterostructures.build(left=hA,right=hB,central=[hC]); ht1.delta=1e-6
ht2=heterostructures.build(left=mirror(hB),right=mirror(hA),central=[mirror(hC)]); ht2.delta=1e-6
for E in [-2.5,-1.9,-1.0,0.6,1.8,2.7,3.0]:
    a=ht1.landauer(energy=E); b=ht2.landauer(energy=E)
    print("E=%+.2f  T=%.6f   T_mirror=%.6f   diff=%.2e"%(E,a,b,abs(a-b)))
```

## `p6_dysonNNN.py`

Backs: **L3 transport dagger** — CLEARED: dysonNNN / dysonLR supercell folding with non-Hermitian long-range hoppings

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.greentk.dyson import dysonNNN,dysonLR

n=2
H0=make_H0(seed=2); T1=make_T(seed=1); T2=0.5*make_T(seed=5); T3=0.3*make_T(seed=9)
def finite(hops,N,E,delta):
    # hops[0]=intra, hops[r]=H_{i,i+r}
    M=np.zeros((N*n,N*n),dtype=np.complex128)
    for i in range(N): M[i*n:(i+1)*n,i*n:(i+1)*n]=hops[0]
    for r in range(1,len(hops)):
        for i in range(N-r):
            M[i*n:(i+1)*n,(i+r)*n:(i+r+1)*n]=hops[r]
            M[(i+r)*n:(i+r+1)*n,i*n:(i+1)*n]=hops[r].conj().T
    G=np.linalg.inv((E+1j*delta)*np.identity(N*n)-M)
    return G[0:n,0:n],G[(N//2)*n:(N//2+1)*n,(N//2)*n:(N//2+1)*n]
E,delta=0.23,0.05
print("--- NNN (dysonNNN) ---")
gs_ref,gb_ref=finite([H0,T1,T2],400,E,delta)
gb,gs=dysonNNN(H0,T1,T2,energy=E,delta=delta)
gb=np.array(gb);gs=np.array(gs)
print("surf err:",np.max(np.abs(gs-gs_ref)),"  bulk err:",np.max(np.abs(gb-gb_ref)))
print("--- LR up to 3rd (dysonLR) ---")
gs_ref3,gb_ref3=finite([H0,T1,T2,T3],400,E,delta)
gb3,gs3=dysonLR([H0,T1,T2,T3],energy=E,delta=delta)
gb3=np.array(gb3);gs3=np.array(gs3)
print("surf err:",np.max(np.abs(gs3-gs_ref3)),"  bulk err:",np.max(np.abs(gb3-gb_ref3)))
print("--- LR with only NN (should reduce) ---")
gsr,gbr=finite([H0,T1],400,E,delta)
gb1,gs1=dysonLR([H0,T1],energy=E,delta=delta)
print("surf err:",np.max(np.abs(np.array(gs1)-gsr)),"  bulk err:",np.max(np.abs(np.array(gb1)-gbr)))
```

## `p7_bloch2d.py`

Backs: **L3 transport dagger** — CLEARED: bloch_selfenergy modes 'full', 'renormalization' and 'adaptive' in 2D

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import make_T,make_H0
from pyqula import geometry
from pyqula.multihopping import MultiHopping
from pyqula.green import bloch_selfenergy

n=2
H0=make_H0(seed=2); Tx=make_T(seed=1); Ty=0.7*make_T(seed=4); Txy=0.35*make_T(seed=6)
g=geometry.square_lattice()
g=g.get_supercell([1,1]) # keep 1 site, build 2 orbitals by spinful trick? no
h=g.get_hamiltonian(has_spin=True)  # 2x2 per site -> use spin as orbital index
d={(0,0,0):np.array(H0,dtype=np.complex128),
   (1,0,0):np.array(Tx,dtype=np.complex128),
   (-1,0,0):np.array(Tx).conj().T,
   (0,1,0):np.array(Ty,dtype=np.complex128),
   (0,-1,0):np.array(Ty).conj().T,
   (1,1,0):np.array(Txy,dtype=np.complex128),
   (-1,-1,0):np.array(Txy).conj().T}
h.set_multihopping(MultiHopping(d))
E,delta=0.17,0.08
hk=h.get_hk_gen()
# brute-force bulk Green's function
nk=400
acc=np.zeros((n,n),dtype=np.complex128)
for ikx in range(nk):
  for iky in range(nk):
    k=[ikx/nk,iky/nk,0.]
    acc+=np.linalg.inv((E+1j*delta)*np.identity(n)-np.array(hk(k)))
acc/=nk*nk
print("brute-force bulk G:\n",acc)
for mode in ["full","renormalization","adaptive","full_adaptive"]:
    try:
        g1,s1=bloch_selfenergy(h,energy=E,delta=delta,mode=mode,nk=200,error=1e-6)
        g1=np.array(g1)
        print(mode,"  maxerr vs brute:",np.max(np.abs(g1-acc)))
    except Exception as e:
        print(mode,"RAISED",type(e).__name__,e)
```

## `p8_embedding.py`

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula import embedding
n=2
T=make_T(seed=1);H0=make_H0(seed=2)
h=build_h(T,H0)
E,delta=0.23,0.05
hk=h.get_hk_gen()
nk=4000
def bulkG(nsuper):
    N=n*nsuper
    acc=np.zeros((N,N),dtype=np.complex128)
    for ik in range(nk):
        k=ik/nk
        # supercell Bloch Hamiltonian
        M=np.zeros((N,N),dtype=np.complex128)
        for i in range(nsuper): M[i*n:(i+1)*n,i*n:(i+1)*n]=H0
        for i in range(nsuper-1):
            M[i*n:(i+1)*n,(i+1)*n:(i+2)*n]=T
            M[(i+1)*n:(i+2)*n,i*n:(i+1)*n]=T.conj().T
        M[(nsuper-1)*n:,0:n]+=T.conj().T*np.exp(-1j*2*np.pi*k)
        M[0:n,(nsuper-1)*n:]+=T*np.exp(1j*2*np.pi*k)
        acc+=np.linalg.inv((E+1j*delta)*np.identity(N)-M)
    return acc/nk
for ns in [1,3,4]:
    eo=embedding.Embedding(h)   # pristine, no defect
    g=eo.get_gf(energy=E,delta=delta,nsuper=ns,nk=600)
    ref=bulkG(ns)
    print("nsuper=%d  max|G_embed - G_bulk| = %.3e   (|G|~%.3f)"%(ns,np.max(np.abs(np.array(g)-ref)),np.max(np.abs(ref))))
```

## `p8b.py`

Backs: **L3 transport dagger** — CLEARED: Embedding.get_gf / dyson.dyson supercell Green's function looked 35% wrong for nsuper>1 with a non-Hermitian T — my reference was wrong, not the code

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
from pyqula import embedding
n=2
E,delta=0.23,0.05
def run(T,H0,tag):
    h=build_h(T,H0)
    nk=8000
    def bulkG(nsuper):
        N=n*nsuper
        acc=np.zeros((N,N),dtype=np.complex128)
        for ik in range(nk):
            k=ik/nk
            M=np.zeros((N,N),dtype=np.complex128)
            for i in range(nsuper): M[i*n:(i+1)*n,i*n:(i+1)*n]=H0
            for i in range(nsuper-1):
                M[i*n:(i+1)*n,(i+1)*n:(i+2)*n]=T
                M[(i+1)*n:(i+2)*n,i*n:(i+1)*n]=T.conj().T
            M[(nsuper-1)*n:,0:n]+=T*np.exp(1j*2*np.pi*k)
            M[0:n,(nsuper-1)*n:]+=T.conj().T*np.exp(-1j*2*np.pi*k)
            acc+=np.linalg.inv((E+1j*delta)*np.identity(N)-M)
        return acc/nk
    for ns in [1,2,3]:
      for nkk in [600,3000]:
        eo=embedding.Embedding(h)
        g=np.array(eo.get_gf(energy=E,delta=delta,nsuper=ns,nk=nkk))
        ref=bulkG(ns)
        print("%s nsuper=%d nk=%d  err=%.3e"%(tag,ns,nkk,np.max(np.abs(g-ref))))
T=make_T(seed=1);H0=make_H0(seed=2)
run(T,H0,"NONHERM")
Th=0.5*np.array([[1.0,0.3],[0.3,0.8]]);   # real symmetric -> hermitian intercell
run(Th,np.array([[0.1,0.0],[0.0,-0.1]]),"HERM   ")
```

## `p9_dyson_ref.py`

Backs: **L3 transport dagger** — CLEARED: Embedding.get_gf / dyson.dyson supercell Green's function looked 35% wrong for nsuper>1 with a non-Hermitian T — my reference was wrong, not the code

```python
import numpy as np,sys
sys.path.insert(0,"SCRATCH")
from fixture import *
n=2;E,delta=0.23,0.05
T=make_T(seed=1);H0=make_H0(seed=2)
# --- oracle 1: huge real-space periodic ring, exact inversion
N=500
M=np.zeros((N*n,N*n),dtype=np.complex128)
for i in range(N): M[i*n:(i+1)*n,i*n:(i+1)*n]=H0
for i in range(N):
    j=(i+1)%N
    M[i*n:(i+1)*n,j*n:(j+1)*n]+=T
    M[j*n:(j+1)*n,i*n:(i+1)*n]+=T.conj().T
G=np.linalg.inv((E+1j*delta)*np.identity(N*n)-M)
def blk(a,b): return G[a*n:(a+1)*n,b*n:(b+1)*n]
print("REAL-SPACE ORACLE")
print("G[0,0]=\n",blk(0,0))
print("G[0,1]=\n",blk(0,1))
print("G[1,0]=\n",blk(1,0))
# --- pyqula dyson supercell
from pyqula.dyson import dyson
h=build_h(T,H0)
g=np.array(dyson(h,[3,1],2000,E+1j*delta))
print("\nPYQULA dyson nsuper=3")
print("g[0,0]=\n",g[0:n,0:n])
print("g[0,1]=\n",g[0:n,n:2*n])
print("g[1,0]=\n",g[n:2*n,0:n])
print("\nerr g00:",np.max(np.abs(g[0:n,0:n]-blk(0,0))))
print("err g01 vs G[0,1]:",np.max(np.abs(g[0:n,n:2*n]-blk(0,1))))
print("err g01 vs G[1,0]:",np.max(np.abs(g[0:n,n:2*n]-blk(1,0))))
print("err g10 vs G[1,0]:",np.max(np.abs(g[n:2*n,0:n]-blk(1,0))))
print("err g10 vs G[0,1]:",np.max(np.abs(g[n:2*n,0:n]-blk(0,1))))
```
