"""Smooth Bloch gauge for the electron-hole pair basis.

algebra.eigh returns each Bloch eigenvector with an arbitrary phase (and,
inside a degenerate multiplet, an arbitrary unitary), so C^{n,k} is a
discontinuous function of k even where the physics is perfectly smooth.
Nothing in the dense BSE cares -- the spectrum is invariant, the gauge
being a block-diagonal unitary on the pair index -- but the quantics
tensor-train solver cares completely: a discontinuous function has no
low-rank quantics representation, and in the raw gauge the BSE kernel is
exactly incompressible.

Measured on this codebase, maximum tensor-train rank of the kernel's
factor tensor at tolerance 1e-6, as the mesh grows 16x:

  1D ionic chain, spinless      npair    128 ->   512 ->  2048
      raw gauge                            16 ->    32 ->    64
      "phase"                               8 ->     8 ->     8
  2D honeycomb, spinless        npair   1024 ->  4096 -> 16384
      raw gauge                            96 ->   192 ->   383
      "phase"                              57 ->    62 ->    63
  2D honeycomb, spinful         npair   4096 -> 16384 -> 65536
      raw gauge                           256 ->   512 ->  1024
      "phase"                             256 ->   512 ->  1024
      "projection"                        182 ->   248 ->   274

The last block is the point of having two modes. Every band of a spinful
Hamiltonian with no spin-orbit coupling and no magnetic order is two-fold
degenerate, and a phase fix cannot help there: what is arbitrary inside a
degenerate subspace is a full unitary, not a phase, so fixing phases
leaves the mixing untouched and the rank saturated -- note the "phase"
row there reproduces the raw-gauge numbers exactly. Projecting the
subspace onto fixed trial orbitals does fix it, which is why
"projection" is what solver="qtt" turns on by default. "phase" is kept
because it is cheaper and is what the non-degenerate measurements above
were made with.

Both modes are k-LOCAL -- each k-point is gauged using only its own
eigenvectors -- which is what lets bsetk/oracle.py apply them inside a
tensor-cross-interpolation oracle that only ever visits O(polylog nk)
k-points. A mesh-global smoothing (parallel transport, or a full
Wannierization through wanniertk/) would need every k-point and would put
the O(nk) scaling straight back.

The projection construction is the first step of Wannier90's disentangled
projection, U = A (A^dag A)^{-1/2} with A_nm = <psi_n|g_m>; see
Miyake and Aryasetiawan, arXiv:0710.4013, for the same object in the
screened-interaction context this package already cites.

THE GAUGE HAS NO PHYSICAL CONTENT. Applying it must leave every exciton
energy unchanged, and tests/bse/test_bse_gauge.py asserts exactly that
against the dense solver. If a gauge choice ever changes a spectrum, it is
a bug in the gauge code, not a modelling decision.
"""
import numpy as np


def fix_gauge(ck,groups,mode="phase",trials=None,refs=None):
    """Return a gauge-fixed copy of ck, shape (nk,nband,norb) with
    ck[ik][n] = C^{n,k}.

    groups is a list of band-index lists, each gauged as one subspace: for
    mode="phase" they are gauged band by band regardless, for
    mode="projection" each group is rotated as a block. Pass the valence
    and conduction windows as two groups.

    trials (mode="projection") is a list of (norb,len(group)) matrices of
    trial orbitals, one per group; refs (mode="phase") is one reference
    orbital index per band. Both default to picking themselves off ck --
    but a caller that gauges different sets of k-points in separate calls
    MUST pass them explicitly and identically, or the two sets are
    smoothed towards different references and the result is not a single
    smooth gauge. That is why bsetk/oracle.py fixes them once on a coarse
    submesh and reuses them for every k-point it later visits."""
    if mode in (None,"none"): return np.array(ck,dtype=np.complex128)
    if mode=="phase":
        if refs is None: refs = default_refs(ck)
        return _phase_gauge(ck,refs)
    if mode=="projection":
        if trials is None: trials = default_trials(ck,groups)
        return _projection_gauge(ck,groups,trials)
    raise ValueError("gauge must be 'phase', 'projection' or None, got %r"
            %(mode,))


def _phase_gauge(ck,refs):
    """Make one reference component of every eigenvector real and
    positive.

    A reference orbital that vanishes somewhere on the mesh puts a branch
    cut back in and the rank with it, so default_refs picks the
    heaviest-weight orbital of each band. When it does vanish anyway the
    phase is left alone at that k-point rather than divided by zero; that
    costs rank locally instead of producing a NaN."""
    out = np.array(ck,dtype=np.complex128)
    for n in range(out.shape[1]):
        ph = out[:,n,int(refs[n])]
        a = np.abs(ph)
        ph = np.where(a>1e-10,ph/np.where(a>0.,a,1.),1.+0.j)
        out[:,n,:] = out[:,n,:]/ph[:,None]
    return out


def default_refs(ck):
    """One reference orbital per band: the one carrying the most weight
    over the k-points given."""
    return [int(np.argmax(np.mean(np.abs(ck[:,n,:]),axis=0)))
            for n in range(ck.shape[1])]


def _projection_gauge(ck,groups,trials):
    """Rotate each band subspace onto fixed trial orbitals.

    Per k-point and per group: A_nm = <psi_n|g_m>, then U = A(A^dag A)^-1/2
    computed as u@vh from the SVD of A, and the new states are
    |w_m> = sum_n |psi_n> U_nm. U is unitary, so the subspace and the
    spectrum are untouched; only the labelling of states inside it moves,
    and it moves to whatever is closest to the trial orbitals, which is a
    smooth function of k as long as A stays invertible.

    A near-singular A means the trial orbitals have no weight on the
    subspace somewhere on the mesh, and the gauge it produces there is
    arbitrary again -- so it is warned about with the offending overlap
    named, rather than silently returning a badly conditioned rotation."""
    import warnings
    out = np.array(ck,dtype=np.complex128)
    worst = np.inf
    for grp,trial in zip(groups,trials):
        grp = list(grp)
        trial = np.array(trial,dtype=np.complex128)
        if trial.shape[1]!=len(grp):
            raise ValueError("need one trial orbital per band of the "
                "group: group of %d bands got %d trials"
                %(len(grp),trial.shape[1]))
        for ik in range(out.shape[0]):
            P = out[ik][grp] # (nb,norb), P[n,a] = C^{n,k}_a
            A = np.conj(P)@trial # <psi_n|g_m>
            u,s,vh = np.linalg.svd(A)
            worst = min(worst,s[-1])
            out[ik][grp] = (u@vh).T@P
    if worst<1e-3:
        warnings.warn("the projection gauge is badly conditioned: the "
            "smallest overlap between a band subspace and its trial "
            "orbitals is %.2e somewhere on the mesh, so the gauge is "
            "nearly arbitrary there and the tensor-train rank will suffer. "
            "Pass trials= explicitly with orbitals that carry weight on "
            "the bands everywhere in the Brillouin zone"%worst,stacklevel=3)
    return out


def default_trials(ck,groups):
    """Pick trial orbitals automatically: for each band subspace, the
    orbitals carrying the largest mesh-averaged weight on it.

    Crude but effective for the models this is used on, where a valence
    subspace sits on one sublattice and a conduction subspace on the
    other. It is a starting gauge, not a Wannier minimization -- if the
    rank does not saturate, an explicit trials= is the first knob."""
    norb = ck.shape[2]
    out = []
    for grp in groups:
        grp = list(grp)
        w = np.mean(np.sum(np.abs(ck[:,grp,:])**2,axis=1),axis=0) # (norb,)
        pick = np.argsort(-w)[0:len(grp)]
        t = np.zeros((norb,len(grp)),dtype=np.complex128)
        for j,a in enumerate(pick): t[a,j] = 1.
        out.append(t)
    return out
