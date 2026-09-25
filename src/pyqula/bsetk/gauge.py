"""Smooth Bloch gauge for the electron-hole pair basis.

algebra.eigh returns each Bloch eigenvector with an arbitrary phase (and,
inside a degenerate multiplet, an arbitrary unitary), so C^{n,k} is a
discontinuous function of k even where the physics is perfectly smooth.
Nothing in the dense BSE cares -- the spectrum is invariant under any
unitary that stays inside a degenerate multiplet, because it commutes
with the diagonal of band energies the pair basis keeps -- but the quantics
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
multiplet onto fixed trial orbitals does fix it, which is why
"projection" is what solver="qtt" turns on by default. "phase" is kept
because it is cheaper and is what the non-degenerate measurements above
were made with.

The projection rotates ONLY inside a degenerate multiplet, never across
a whole valence or conduction window. The pair basis keeps the one-body
part as the diagonal e_c(k+Q) - e_v(k) on the band labels, and a
rotation that mixes bands of different energy makes <w_n|H(k)|w_m>
non-diagonal, so the matrix diagonalized is no longer a unitary
transform of the BSE block. It once did rotate the whole window, which
went unseen on every model whose windows are degenerate or one band
wide, and on a spinful chain with Rashba coupling and a generic exchange
field it moved the lowest exciton by 0.014. On a non-degenerate band the
projection is therefore a phase fix against the band's own trial
orbital, and the rank it gives there is the rank of "phase".

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
energy unchanged, and tests/bse/test_bse_gauge.py and
test_bse_gauge_multiplet.py assert exactly that against the dense solver,
the second on windows of non-degenerate bands. If a gauge choice ever
changes a spectrum, it is a bug in the gauge code, not a modelling
decision.
"""
import numpy as np


def fix_gauge(ck,groups,mode="phase",trials=None,refs=None,ek=None):
    """Return a gauge-fixed copy of ck, shape (nk,nband,norb) with
    ck[ik][n] = C^{n,k}.

    groups is a list of band-index lists: for mode="phase" they are gauged
    band by band regardless, for mode="projection" each degenerate
    multiplet inside a group is rotated as a block, which needs the band
    energies ek, shape (nk,nband). Pass the valence and conduction windows
    as two groups.

    trials (mode="projection") is a list of (norb,len(group)) matrices of
    trial orbitals, one per group, column j going with the j-th band of
    the group; refs (mode="phase") is one reference
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
        if ek is None:
            raise ValueError("gauge='projection' needs the band energies "
                "ek, because it rotates only inside degenerate multiplets: "
                "a rotation across bands of different energy would change "
                "the exciton spectrum")
        if trials is None: trials = default_trials(ck,groups)
        return _projection_gauge(ck,ek,groups,trials)
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


def _projection_gauge(ck,ek,groups,trials,tol=1e-8):
    """Rotate each degenerate multiplet onto fixed trial orbitals.

    Per k-point, per group and per degenerate multiplet inside the group
    (bands closer than tol in energy at that k): A_nm = <psi_n|g_m> with
    g_m the trial orbitals at the multiplet's positions in the group,
    then U = A(A^dag A)^-1/2 computed as u@vh from the SVD of A, and the
    new states are |w_m> = sum_n |psi_n> U_nm. U is unitary and acts only
    among states of one energy, so it commutes with the band-energy
    diagonal that PairBasis keeps, and the spectrum is untouched. A
    rotation across non-degenerate bands would not be, which is why the
    group is not rotated as one block: the pair basis keeps dE as
    e_c(k+Q) - e_v(k) on the band labels, and that is the one-body part
    only while <w_n|H(k)|w_m> stays diagonal. On a non-degenerate band
    the multiplet is one band, A is a number and U its phase, so this
    reduces to a phase fix with the trial orbital as the reference.

    The trials therefore go with band positions, which is the diagonal
    block of the whole-group rotation. Where two non-degenerate bands of
    one group cross between mesh points, the sorted labels swap and each
    label changes character, so the pair basis has a jump there whatever
    the trials are: it is the price of a diagonal dE, and only a rotation
    across the crossing bands could smooth it, which is exactly what
    changes the spectrum.

    A near-singular A means the trial orbitals have no weight on the
    multiplet somewhere on the mesh, and the gauge it produces there is
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
            for pos in degenerate_groups(ek[ik][grp],tol=tol):
                bands = [grp[j] for j in pos] # one degenerate multiplet
                P = out[ik][bands] # (nb,norb), P[n,a] = C^{n,k}_a
                A = np.conj(P)@trial[:,pos] # <psi_n|g_m>
                u,s,vh = np.linalg.svd(A)
                worst = min(worst,s[-1])
                out[ik][bands] = (u@vh).T@P
    if worst<1e-3:
        warnings.warn("the projection gauge is badly conditioned: the "
            "smallest overlap between a band multiplet and its trial "
            "orbitals is %.2e somewhere on the mesh, so the gauge is "
            "nearly arbitrary there and the tensor-train rank will suffer. "
            "Pass trials= explicitly with orbitals that carry weight on "
            "the bands everywhere in the Brillouin zone"%worst,stacklevel=3)
    return out


def degenerate_groups(es,tol=1e-8):
    """Group the indices of a sorted energy list into degenerate blocks"""
    groups,cur = [],[0]
    for i in range(1,len(es)):
        if abs(es[i]-es[i-1])<tol: cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def default_trials(ck,groups):
    """Pick trial orbitals automatically: for each band subspace, one
    orbital per band, column j of the trial matrix going with the j-th
    band, chosen so the orbitals are distinct and carry the largest total
    mesh-averaged weight on their bands.

    The columns have to follow the bands rather than the subspace as a
    whole, because _projection_gauge rotates a non-degenerate band only
    by a phase against its own column: on a ferromagnetic chain the
    valence subspace sits on one site with both spins, and the two
    heaviest orbitals in the wrong order give each band the orbital of
    the other spin, an overlap of exactly zero and an arbitrary gauge.

    Crude but effective for the models this is used on, where a valence
    subspace sits on one sublattice and a conduction subspace on the
    other. It is a starting gauge, not a Wannier minimization -- if the
    rank does not saturate, an explicit trials= is the first knob."""
    from scipy.optimize import linear_sum_assignment
    norb = ck.shape[2]
    out = []
    for grp in groups:
        grp = list(grp)
        w = np.mean(np.abs(ck[:,grp,:])**2,axis=0) # (nband,norb)
        bands,pick = linear_sum_assignment(w,maximize=True)
        t = np.zeros((norb,len(grp)),dtype=np.complex128)
        for j,a in zip(bands,pick): t[a,j] = 1.
        out.append(t)
    return out
