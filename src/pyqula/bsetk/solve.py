import numpy as np
import scipy.linalg as lg
from .pairbasis import PairBasis
from .interaction import bare_interaction
from .kernel import build_blocks


class BSE():
    """Solution of the Bethe-Salpeter equation at a fixed center-of-mass
    momentum Q, on top of a (mean-field) Hamiltonian.

    screening selects the interaction of the DIRECT (ladder) term:

      None (default)  the bare interaction, i.e. time-dependent
                      Hartree-Fock -- the behavior this class always had
      "rpa"           the static RPA screened interaction W = eps^-1 v,
                      built from the bands of the mean field itself
                      (bsetk/screening.py). This is the GW-BSE-style
                      construction and is what physically binds excitons
      "crpa"          the same, with the transitions inside the nv/nc BSE
                      band window excluded from the polarization
      a ScreenedInteraction  a precomputed one, e.g. to reuse the same W
                      across a scan of Q without rebuilding it

    channel picks where the dielectric matrix is built: "charge" (the
    default, the standard GW construction, spin-rotation invariant) or
    "orbital" (dress the full spin-orbital matrix, which does not
    preserve SU(2)). See bsetk/screening.charge_channel. The two coincide
    for a spinless Hamiltonian.

    The EXCHANGE term always keeps the bare interaction, whatever this is
    set to -- see build_blocks for why.

    solver picks how the eigenproblem is solved:

      "dense" (default)  assemble the matrix and diagonalize it. The only
                  route that gives the FULL (non-Tamm-Dancoff) spectrum
                  and every exciton, and the only one whose behavior is
                  unchanged by this option existing. Its cost is the
                  (2*npair)^2 matrix that check_memory guards.
      "iterative" apply the kernel matrix-free through its exact low-rank
                  factorization (bsetk/factorize.py) and run LOBPCG for
                  the neig lowest excitons. No dense matrix ever exists,
                  so the memory wall is gone; Tamm-Dancoff only.
      "qtt"       quantics tensor train: compress the kernel into an MPO
                  and solve it with DMRG (bsetk/qtt.py). Cost grows with
                  log(nk) rather than nk, at the price of a smooth-gauge
                  requirement, a tensor-train tolerance and a dmrgpy
                  dependency. Tamm-Dancoff only.

    neig is how many excitons solver="iterative" returns; "dense" returns
    all of them and ignores it, and "qtt" accepts only neig=1. The default
    None means 4 for "iterative" and 1 for "qtt".

    gauge smooths the arbitrary phase algebra.eigh leaves on each Bloch
    eigenvector (bsetk/gauge.py). It changes no energy -- it is a
    block-diagonal unitary on the pair index -- so the default "auto"
    turns it on only for solver="qtt", which cannot work without it, and
    leaves the other two in the raw gauge where it makes no difference.
    Setting it explicitly ("phase", "projection" or None) applies to every
    solver, which is how the invariance is checked on the dense one.

    BEFORE TURNING SCREENING ON, read bsetk/screening.py's module
    docstring. In short: a Hubbard U fitted to a material is already an
    effective screened interaction and must not be screened again, and the
    default V=h.V from a Hubbard SCF is exactly that case. Screening is
    for a genuinely bare interaction, i.e. a long-range Coulomb tail from
    interaction.density_interaction(Vr=...).

    Attributes after solving:
      energies    exciton energies, the positive eigenvalues, sorted
      amplitudes  (nexciton,npair) resonant amplitudes A_{vc}(k)
      amplitudesY (nexciton,npair) antiresonant amplitudes; identically
                  zero under the Tamm-Dancoff approximation
      pairs       the PairBasis, holding the k-mesh, the band window and
                  the (ik,iv,ic) label of every pair index -- or, for
                  solver="qtt", an oracle.PairOracle, which exposes the
                  same things but computes them per k-point on demand and
                  reports how many it actually diagonalized (.ndiag())
      W           the interaction of the direct term (screened, if asked)
      Wx          the bare interaction, used by the exchange term
    """
    def __init__(self,h,V=None,Q=None,nk=10,nv=None,nc=None,
            kernel="full",tda=False,max_memory=2.0,
            screening=None,nkW=None,channel="charge",
            solver="dense",neig=None,gauge="auto",**kwargs):
        if solver not in ("dense","iterative","qtt"):
            raise ValueError("solver must be 'dense', 'iterative' or "
                    "'qtt', got %r"%(solver,))
        if solver!="dense" and not tda:
            # not a silent switch: the two answers differ, and which one
            # was computed has to be unambiguous
            raise ValueError("solver=%r solves the Tamm-Dancoff problem "
                "only, so it needs tda=True. The full (non-Tamm-Dancoff) "
                "BSE matrix is not Hermitian and is diagonalized through a "
                "Cholesky factorization of S@H (see "
                "solve_pseudo_hermitian), which needs the dense matrix; "
                "large-scale BSE codes use the Tamm-Dancoff approximation "
                "for the same reason. tests/bse/test_bse_physics.py shows "
                "the two agreeing at weak coupling, which is the regime "
                "bound excitons live in"%(solver,))
        if neig is None:
            # the quantics solver returns the lowest exciton only (see
            # qtt.solve_qtt for the measurements behind that), so the
            # default cannot be the same for both
            neig = 1 if solver=="qtt" else 4
        if gauge=="auto":
            # the gauge is unobservable, so only the solver that needs it
            # pays for it: "qtt" cannot work without a smooth gauge, the
            # other two are exact in any gauge. An explicit gauge= is
            # honored everywhere, which is what makes the invariance
            # testable on the dense solver
            gauge = "projection" if solver=="qtt" else None
        self.solver = solver
        self.neig = neig
        self.gauge = gauge
        if solver=="qtt": # never builds the mesh-wide pair basis
            from .qtt import solve_qtt
            # note PairBasis is deliberately not constructed above: it
            # diagonalizes every point of the mesh, which is the one thing
            # this solver exists not to do
            self.pairs,self.W,self.Wx,out = solve_qtt(h,V=V,Q=Q,nk=nk,
                    nv=nv,nc=nc,kernel=kernel,neig=neig,gauge=gauge,
                    screening=screening,nkW=nkW,channel=channel,**kwargs)
            self.screening = screening
            self.kernel = kernel
            self.tda = tda
            self.A = self.Abar = self.B = None
            self.energies,self.amplitudes = out
            self.amplitudesY = np.zeros(self.amplitudes.shape,
                    dtype=np.complex128)
            return
        self.pairs = PairBasis(h,Q=Q,nk=nk,nv=nv,nc=nc,gauge=gauge)
        self.Wx = bare_interaction(h,V=V) # bare, for the exchange term
        self.W = get_direct_interaction(h,self.pairs,V=V,nk=nk,
                screening=screening,nkW=nkW,kernel=kernel,channel=channel)
        self.screening = screening
        self.kernel = kernel
        self.tda = tda
        if solver=="iterative":
            from .iterative import solve_iterative
            self.A = self.Abar = self.B = None
            es,ws = solve_iterative(self.pairs,self.W,Wx=self.Wx,
                    kernel=kernel,neig=neig,**kwargs)
            self.energies = es.astype(np.complex128)
            self.amplitudes = ws
            self.amplitudesY = np.zeros(ws.shape,dtype=np.complex128)
            return
        check_memory(self.pairs.npair,tda=tda,max_memory=max_memory)
        self.A,self.Abar,self.B = build_blocks(self.pairs,self.W,
                Wx=self.Wx,kernel=kernel)
        self.solve()
    def _get_matrix(self):
        """Return the full BSE matrix that is diagonalized,

            [[   A   ,     B      ],
             [ -B^dag, -conj(Abar)]]

        NOTE the lower-left block is -B^dag, not the -B^* the
        Casida/linear-response equations are usually written with. Those
        two agree only when B is symmetric, which holds at Q=0 (where the
        resonant and antiresonant pair sets coincide) but NOT at finite Q,
        where B connects two different sets -- pairs (v,k)->(c,k+Q) on one
        side and (c,k)->(v,k+Q) on the other. Using -B^* there silently
        gives eigenvalues that are not the BSE poles at all, while still
        looking perfectly plausible; tests/bse/test_bse_rpa.py is what
        catches it, since the exchange-only cross-check against chitk.rpa
        passes at Q=0 and fails at Q!=0 with the conjugate. Likewise the
        lower-right block is built from the antiresonant block Abar, not
        from A -- at finite Q the antiresonant transition energies
        e_c(k)-e_v(k+Q) differ from the resonant ones e_c(k+Q)-e_v(k).

        With A and Abar Hermitian this makes diag(1,-1) @ H Hermitian, so
        the spectrum is real and comes in +-E pairs as long as the
        mean-field reference is stable against this excitation."""
        if self.tda: return self.A
        return np.block([[self.A,self.B],
                         [-self.B.conj().T,-np.conj(self.Abar)]])
    def solve(self):
        """Diagonalize the BSE matrix and keep the positive branch"""
        if self.tda: # Tamm-Dancoff: the resonant block alone, Hermitian
            es,ws = lg.eigh(self.A)
            self.energies = es
            self.amplitudes = np.array(ws.T,dtype=np.complex128)
            self.amplitudesY = np.zeros(self.amplitudes.shape,
                    dtype=np.complex128)
            return
        es,ws = solve_pseudo_hermitian(self.get_matrix())
        np_ = self.pairs.npair
        # The spectrum comes in +-E pairs: +E are the excitons at +Q, -E
        # the ones at -Q. Keep the positive branch. An eigenvalue with a
        # sizable imaginary part means the mean-field reference is
        # unstable against this excitation (the interaction has overcome
        # the gap), which is physically meaningful, so it is reported
        # rather than discarded -- but it is sorted by its real part.
        keep = es.real>0.
        es,ws = es[keep],ws[keep]
        order = np.argsort(es.real)
        es,ws = es[order],ws[order]
        if np.max(np.abs(es.imag))<1e-8*max(1.,np.max(np.abs(es.real))):
            es = es.real+0.0j # clean up numerical noise
        X,Y = ws[:,0:np_],ws[:,np_:]
        norm = np.sum(np.abs(X)**2,axis=1)-np.sum(np.abs(Y)**2,axis=1)
        # X^dag X - Y^dag Y is the conserved norm of the linear-response
        # problem, not X^dag X + Y^dag Y; it is positive on the positive
        # branch of a stable reference state
        safe = np.abs(norm)>1e-12
        scale = np.ones(len(norm))
        scale[safe] = 1./np.sqrt(np.abs(norm[safe]))
        self.energies = es
        self.amplitudes = X*scale[:,None]
        self.amplitudesY = Y*scale[:,None]
    def get_energies(self,n=None):
        """Return the exciton energies, optionally only the n lowest"""
        es = self.energies
        if np.max(np.abs(es.imag))<1e-10: es = es.real
        if n is None: return es
        return es[0:n]
    def get_binding_energies(self,n=None):
        """Return the binding energies, i.e. how far below the lowest
        independent-particle transition on this mesh each exciton lies.
        Positive means bound."""
        return self.get_lowest_transition() - self.get_energies(n=n)
    def get_lowest_transition(self):
        """The lowest independent-particle transition on this mesh.

        A minimum over the mesh for the solvers that hold it; for the
        quantics solver, whose pair basis is never materialized, the same
        number computed as the ground state of the diagonal MPO alone --
        which is exactly what kernel="none" means, and stays logarithmic
        in the mesh."""
        if self.solver=="qtt": return self.pairs.lowest_transition
        return np.min(self.pairs.dE)
    def get_matrix(self):
        if self.A is None:
            raise ValueError("solver=%r never builds the BSE matrix -- "
                "that is the point of it. Use solver='dense' if the "
                "matrix itself is what is wanted"%(self.solver,))
        return self._get_matrix()


def get_direct_interaction(h,pb,V=None,nk=10,screening=None,nkW=None,
        kernel="full",channel="charge"):
    """Return the interaction the direct (ladder) term should use.

    screening=None gives the bare interaction back, so nothing about the
    old behavior changes. Otherwise the static RPA screened interaction is
    built on the mesh -- reusing the PairBasis's own diagonalizations when
    the screening mesh is the BSE mesh, which is the common case and makes
    the screening almost free next to the kernel build."""
    import warnings
    if screening is None: return bare_interaction(h,V=V)
    if hasattr(screening,"at"): return screening # precomputed, reuse it
    from .screening import screened_interaction
    if kernel in ("exchange","none"):
        warnings.warn("screening=%r has no effect with kernel=%r: the "
            "screened interaction only enters the direct (ladder) term, "
            "and the exchange term deliberately keeps the bare "
            "interaction. The result is the unscreened one"
            %(screening,kernel),stacklevel=3)
    nkW = nk if nkW is None else nkW
    reuse = (nkW==nk) # can the pair basis's own eigenstates be reused?
    if not reuse:
        # A Gamma-centered mesh of m*nk points contains every point of the
        # nk one, so the q-points the direct term asks for are still
        # tabulated; any other ratio leaves holes and is refused rather
        # than interpolated over.
        if nkW%nk!=0 or nkW<nk:
            raise ValueError("nkW = %d must be a positive integer multiple "
                "of nk = %d. The direct term needs W at the differences of "
                "the BSE k-mesh, and only a mesh whose density is a "
                "multiple of it contains those points; anything else would "
                "have to be interpolated, which is not done here"%(nkW,nk))
    exclude = (pb.vbands,pb.cbands) if screening=="crpa" else None
    return screened_interaction(h,V=V,nk=nkW,screening=screening,
            exclude=exclude,channel=channel,
            kpoints=pb.kpoints if reuse else None,
            ek=pb.ek if reuse else None,
            ck=pb.ck if reuse else None)


def solve_pseudo_hermitian(H):
    """Diagonalize the BSE matrix, exploiting its structure.

    H itself is not Hermitian, but S@H with S = diag(1,...,1,-1,...,-1) is
    (that is what the block layout in BSE.get_matrix buys). Handing H
    straight to a general non-Hermitian eigensolver therefore throws that
    structure away and pays for it in accuracy: H is non-normal, so its
    eigenvalue errors scale with the conditioning of its eigenvector
    matrix, and on a degenerate spectrum that costs several digits
    (measured: ~5e-5 absolute error on a 128x128 BSE whose exact answer is
    known by supercell folding, against ~3e-14 for the route below).

    So instead: write K = S@H, which is Hermitian and, for a mean-field
    reference that is stable against every excitation, positive definite.
    Cholesky it, K = L@L^dag. Then H = S@K is similar to L^dag@S@L, which
    is Hermitian, so eigh gives its eigenvalues to machine precision, and
    the eigenvectors come back as x = L^-dag y.

    If the Cholesky fails, K is not positive definite: the reference state
    is unstable against some excitation (the interaction has overcome the
    gap) and that excitation's energy is genuinely imaginary. That is
    physically meaningful rather than an error, so fall back to the
    general solver and let the caller see the complex eigenvalue."""
    n = H.shape[0]//2
    S = np.concatenate([np.ones(n),-np.ones(n)])
    K = S[:,None]*H # S@H, S diagonal
    K = (K+K.conj().T)/2. # symmetrize away roundoff
    try:
        L = lg.cholesky(K,lower=True)
    except lg.LinAlgError:
        es,ws = lg.eig(H)
        return es,np.array(ws.T,dtype=np.complex128)
    M = (L.conj().T*S[None,:])@L # L^dag @ S @ L, Hermitian (S is diagonal)
    M = (M+M.conj().T)/2.
    es,ys = lg.eigh(M)
    # x = L^-dag y, i.e. solve L^dag x = y
    xs = lg.solve_triangular(L.conj().T,ys,lower=False)
    return es.astype(np.complex128),np.array(xs.T,dtype=np.complex128)


def check_memory(npair,tda=False,max_memory=2.0):
    """Raise before allocating if the dense BSE matrix would not fit.

    The BSE matrix is (2*npair)^2 complex128 for the full problem (npair^2
    under the Tamm-Dancoff approximation), but that is not the footprint:
    assembling it, symmetrizing it, factorizing it and running the dense
    eigensolver each need working copies on top. The multipliers below are
    measured rather than guessed -- a 3200x3200 full BSE matrix is 164 MB
    on its own and was observed to cost about 1.5 GB of resident memory
    end to end, i.e. roughly eight times the matrix.

    npair = nv*nc*nk grows fast with the k-mesh, which is exactly the knob
    users turn first, so failing here with the numbers spelled out beats
    an opaque MemoryError (or a machine that starts swapping) later."""
    n = npair if tda else 2*npair
    gb = n*n*16/1e9 # complex128
    total = gb*(4. if tda else 8.) # plus the eigensolver's working copies
    if total>max_memory:
        raise MemoryError("this BSE calculation needs about %.1f GB "
            "(a %d x %d dense matrix from npair = %d electron-hole pairs, "
            "plus eigensolver workspace), above the max_memory = %.1f GB "
            "limit. Reduce nk, narrow the band window with nv/nc, use "
            "tda=True (4x smaller), or raise max_memory"
            %(total,n,n,npair,max_memory))
