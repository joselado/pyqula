"""Static RPA screened interaction W = eps^-1 v, built from the bands of a
mean-field Hamiltonian.

The BSE kernel binds an electron-hole pair with an interaction. Using the
*bare* one (the default, i.e. h.V) makes the whole thing time-dependent
Hartree-Fock, which is internally consistent but overestimates the
attraction badly in anything but a very small system: what actually binds
an exciton is the interaction screened by the polarizability of the very
bands the mean field just produced. Xatu (arXiv:2307.01572), the code this
BSE implementation follows, does not compute that -- it substitutes a
phenomenological Rytova-Keldysh form. Since the mean-field step already
hands us {e_n(k), C^n(k)} on a k-mesh, computing the RPA W instead of
postulating it is a small amount of extra work.

Everything here stays in the same point-like-orbital convention and cell
gauge the rest of bsetk/ uses (see interaction.interaction_at_q for why no
intracell-position phases appear), so the density form factor of a
transition is diagonal in the orbital index,

    rho^{nm}_a(k,q) = conj(C^{n,k}_a) C^{m,k+q}_a

-- the very object kernel.exchange_block already consumes as
conj(pb.el)*pb.ho. The static Adler-Wiser polarizability is then a sum of
outer products of those length-norb vectors,

    chi0_ab(q) = (1/N) sum_k sum_{n,m} (f_nk - f_{m,k+q})
                     * rho^{nm}_a(k,q) conj(rho^{nm}_b(k,q))
                     / (e_nk - e_{m,k+q})

and per q

    eps_ab(q) = delta_ab - sum_c v_ac(q) chi0_cb(q)
    W(q)      = eps^-1(q) v(q)

The reference for a screened interaction evaluated in a localized basis
(and for the constrained-RPA variant) is Miyake and Aryasetiawan,
arXiv:0710.4013.

TWO THINGS THAT ARE EASY TO GET WRONG
-------------------------------------

**Both orderings of (n,m) must be summed.** The familiar "twice one
ordering" shortcut -- sum only over (n occupied, m empty) and multiply by
two -- is valid only with time-reversal symmetry, and this codebase's SCF
routinely converges to magnetic mean fields where it is false. The
unrestricted double sum with the (f_n - f_m) factor above is what is
implemented: the occupation difference kills the same-occupancy terms by
itself, so both orderings survive and nothing is double counted.

That is not pedantry, because it is what keeps kernel.build_blocks valid.
That function builds the antiresonant exchange block from
WmQ = np.conj(WQ), which holds for a real-space interaction with real
entries. A numerically tabulated W(q) has no such guarantee a priori --
but with both orderings summed, chi0(q) is Hermitian and satisfies
chi0_ab(q) = chi0_ba(-q), i.e. chi0(-q) = conj(chi0(q)), and carrying that
through the inversion with a Hermitian v gives W(-q) = conj(W(q)) again.
tests/bse/test_bse_screening.py asserts this on a *magnetic* model, which
is the case that separates it from the time-reversal-symmetric one.

**A fitted Hubbard U must not be screened.** A U chosen to reproduce a
material is already an effective, screened interaction; running it through
this module screens it a second time and gives a badly underbound exciton.
This is for a genuinely *bare* interaction -- a long-range Coulomb tail
entered through interaction.density_interaction(Vr=...), or bare model
V1/V2/V3 shells. The dangerous case is exactly the default one, V=h.V from
a Hubbard SCF.

**The RPA here is not spin-rotation invariant.** This is the sharpest
limitation and it bites on the most ordinary case, a non-magnetic spinful
model. On such a reference the Bloch states are spin-diagonal, so chi0 is
proportional to the identity in spin; the bare interaction's spin
structure is spanned by {1, sigma_x} (a site pair couples every spin
combination equally, a site's own block is the up-down-only Hubbard term),
and that algebra is commutative, so eps and W stay inside it -- but with
the same-spin and opposite-spin entries no longer EQUAL. Writing the
result as

    A n_iu n_ju + B n_iu n_jd + ... = (A+B)/2 n_i n_j + 2(A-B) Sz_i Sz_j

the screened interaction has picked up an Ising Sz-Sz coupling, which is
not SU(2) invariant. The visible consequence is that a spin multiplet of
the exciton spectrum splits: measured on the gapped honeycomb, a lowest
transition that is four-fold degenerate to 1e-14 with the bare
interaction comes out split by 4e-3 once screened
(tests/bse/test_bse_screening.py pins this).

This is an artifact of resumming charge bubbles with a density-density
kernel in the spin-orbital basis, not of the code: RPA in the density
channel generically generates Ising-like effective spin couplings. The
standard fix is the GW one -- build the dielectric matrix in the charge
channel alone, as a matrix over SITE indices, and left-multiply the bare
interaction by it, so its spin structure is untouched by construction.
That is a different approximation with its own convention questions (what
exactly the charge-channel v of an up-down-only Hubbard term is), so it
is deliberately not decided here. Until it is: spinless models are
unaffected, and on a spinful one read the spin structure of the result
with this in mind.

Note also that bubbles inside W and ladders in the BSE are different
diagram classes, so a full-RPA W with a BSE ladder is the standard
GW-BSE construction and is NOT double counting. The constrained variant
(screening="crpa") excludes the transitions inside the BSE band window
from chi0, which is the right choice when that window is being treated as
a downfolded model solved exactly afterwards.
"""

import numpy as np
from numba import jit,prange
from .. import algebra
from .. import parallel
from .interaction import bare_interaction,interaction_at_q,qkey
from .pairbasis import select_bands


class ScreenedInteraction():
    """A screened interaction tabulated at the q-points of a k-mesh.

    Unlike the real-space dictionaries the rest of bsetk/ passes around,
    this object exists only ON the mesh: W(q) is the result of a matrix
    inversion at each q, not the Fourier transform of anything short
    ranged. That is exactly what the BSE direct term needs (see .at), and
    is a hard error for anything else (see the note there).

    Attributes:
      qs      (nq,3) the mesh q-points W is tabulated at
      Wq      (nq,norb,norb) the screened interaction at each of them
      chi0    (nq,norb,norb) the polarizability it was built from
      bare    the bare interaction dictionary, for reference
      epsmin  smallest real part of any eigenvalue of eps(q) over the
              mesh -- how close the RPA came to diverging
    """
    def __init__(self,qs,Wq,geometry,nkmesh,chi0=None,bare=None,
            epsmin=None,screening="rpa"):
        self.qs = np.array(qs,dtype=np.float64)
        self.Wq = np.array(Wq,dtype=np.complex128)
        self.geometry = geometry
        self.nkmesh = np.array(nkmesh,dtype=np.int64)
        self.chi0 = chi0
        self.bare = bare
        self.epsmin = epsmin
        self.screening = screening
        self._index = {qkey(q): i for i,q in enumerate(self.qs)}
    def at(self,q):
        """Return W at a q-point of the mesh.

        Raises on any other q rather than snapping to the nearest mesh
        point. That is not defensiveness for its own sake -- the BSE
        direct term only ever needs W at differences of mesh k-points,
        which are mesh points themselves (kernel.qdifference_map), so on
        the intended path this never fires. What does hit it is handing
        this object back in as a plain V=, because the *exchange* term
        needs W at the center-of-mass momentum Q, and get_exciton_bands
        scans Q along a continuous path. Use the screening= argument
        instead (which keeps the exchange term bare, as it should be), or
        get_dict() for something evaluable at arbitrary q."""
        key = qkey(q)
        if key not in self._index:
            raise ValueError("this screened interaction is tabulated only "
                "on the %s k-mesh it was built from, and q = %s is not a "
                "point of it. A screened W(q) is the inverse of a matrix "
                "evaluated at each mesh q, not the Fourier transform of a "
                "short-ranged real-space interaction, so it cannot simply "
                "be evaluated somewhere else. If this came from passing a "
                "ScreenedInteraction as V= to the BSE, use "
                "screening='rpa' instead: that screens the direct term "
                "(which only ever needs mesh q-points) and correctly "
                "leaves the exchange term bare. If you genuinely need W "
                "off the mesh, use .get_dict() and accept its truncation "
                "error"%(list(self.nkmesh),np.array(q)))
        return self.Wq[self._index[key]]
    def get_dict(self,tol=1e-8):
        """Inverse Fourier transform back to a real-space interaction
        dictionary {(n1,n2,n3): matrix}, in the same convention
        interaction.bare_interaction returns.

        This is the escape hatch from the mesh: the result can be
        evaluated at any q, fed to get_mean_field_hamiltonian(V=...) to
        re-converge the mean field with the screened interaction, or
        simply inspected/plotted to see how far the screened interaction
        reaches.

        It is not free, though. W(q) on an nk-point mesh determines W(d)
        on exactly nk lattice vectors, so the tail beyond the mesh
        supercell is aliased back in. Entries whose largest element is
        below tol are dropped."""
        nq = len(self.qs)
        ds = _mesh_vectors(self.nkmesh,self.geometry.dimensionality)
        out = dict()
        for d in ds:
            m = np.zeros(self.Wq[0].shape,dtype=np.complex128)
            for iq,q in enumerate(self.qs):
                m = m + self.Wq[iq]*np.conj(self.geometry.bloch_phase(d,q))
            m = m/nq
            if np.max(np.abs(m))>tol: out[d] = m
        return out
    def __str__(self):
        return "ScreenedInteraction(%s, %d q-points, min eig eps = %s)"%(
                self.screening,len(self.qs),self.epsmin)


def _mesh_vectors(nkmesh,dim):
    """Lattice vectors dual to a k-mesh, centered around the origin: the
    nk_i values -nk_i//2 ... nk_i-nk_i//2-1 along each periodic direction,
    and 0 along the rest. Centering matters -- the aliased tail of the
    inverse transform is then split symmetrically around the cell instead
    of being dumped entirely on one side."""
    rngs = []
    for i in range(3):
        if i<dim:
            n = int(nkmesh[i])
            rngs.append(list(range(-(n//2),n-(n//2))))
        else: rngs.append([0])
    out = []
    for d0 in rngs[0]:
        for d1 in rngs[1]:
            for d2 in rngs[2]: out.append((d0,d1,d2))
    return out


@jit(nopython=True,parallel=True,cache=True)
def polarizability_jit(ck,ek,occ,ikq,allowed):
    """Static polarizability chi0_ab(q), see the module docstring.

      ck       (nk,nb,norb) Bloch coefficients, ck[ik,n] = C^{n,k}
      ek       (nk,nb) band energies
      occ      (nk,nb) occupations (1 below the Fermi energy, 0 above)
      ikq      (nk,nq) index of the mesh point k_ik + q_iq
      allowed  (nb,nb) False for transitions excluded from the
               polarization (this is what makes cRPA cRPA)

    The (n,m) double sum is unrestricted; the (f_n - f_m) factor kills the
    same-occupancy terms, so both orderings enter exactly once each. See
    the module docstring for why summing only one of them and doubling is
    wrong on a magnetic mean field."""
    nk,nb,norb = ck.shape
    nq = ikq.shape[1]
    out = np.zeros((nq,norb,norb),dtype=np.complex128)
    for iq in prange(nq): # loop over q-points, in parallel
        acc = np.zeros((norb,norb),dtype=np.complex128)
        rho = np.zeros(norb,dtype=np.complex128)
        for ik in range(nk): # Brillouin zone sum
            jk = ikq[ik,iq] # the mesh point k+q
            for n in range(nb): # first band
                for m in range(nb): # second band
                    df = occ[ik,n] - occ[jk,m] # occupation difference
                    if df==0.: continue # same occupancy, no contribution
                    if not allowed[n,m]: continue # excluded (cRPA)
                    w = df/(ek[ik,n] - ek[jk,m])
                    for a in range(norb): # density form factor
                        rho[a] = np.conj(ck[ik,n,a])*ck[jk,m,a]
                    for a in range(norb): # accumulate the outer product
                        ra = w*rho[a]
                        if ra!=0.:
                            for b in range(norb):
                                acc[a,b] += ra*np.conj(rho[b])
        out[iq] = acc/nk # 1/N, N the number of unit cells
    return out


def mesh_sum_index(kpoints,qs):
    """Return ikq[ik,iq], the index of the mesh point kpoints[ik]+qs[iq].

    geometry.get_kmesh returns a Gamma-centered uniform mesh in fractional
    coordinates, so a mesh k-point plus a mesh q-point is again a mesh
    k-point (modulo a reciprocal lattice vector) and the whole
    polarizability needs only ONE diagonalization per mesh point rather
    than one per (k,q) pair."""
    index = {qkey(k): i for i,k in enumerate(kpoints)}
    nk,nq = len(kpoints),len(qs)
    ikq = np.zeros((nk,nq),dtype=np.int64)
    for ik in range(nk):
        for iq in range(nq):
            key = qkey(kpoints[ik]+qs[iq])
            if key not in index:
                raise ValueError("k+q fell off the k-mesh, which should "
                    "not happen on the Gamma-centered uniform mesh "
                    "get_kmesh returns. Was a custom set of k-points "
                    "passed in?")
            ikq[ik,iq] = index[key]
    return ikq


def mesh_eigenstates(h,nk=10):
    """Diagonalize h on the k-mesh, returning (kpoints,ek,ck) in the same
    layout PairBasis stores them in -- so a BSE that already has them can
    hand them straight to static_polarizability instead of paying for a
    second set of diagonalizations."""
    h = h.get_multicell().get_dense()
    kpoints = np.array(h.geometry.get_kmesh(nk=nk),dtype=np.float64)
    hk = h.get_hk_gen()
    eks,cks = [],[]
    for k in kpoints:
        e,w = algebra.eigh(hk(k))
        eks.append(e) ; cks.append(np.array(w.T,dtype=np.complex128))
    return kpoints,np.array(eks),np.array(cks)


def static_polarizability(h,nk=10,exclude=None,kpoints=None,ek=None,ck=None):
    """Return (qs,chi0), the static RPA polarizability of h on its k-mesh.

    chi0 has shape (nq,norb,norb) and qs are the mesh points themselves --
    which are exactly the q-points the BSE direct term asks for, since
    kernel.qdifference_map's distinct k-k' differences of a
    Gamma-centered mesh are mesh points.

    exclude, if given, is a (vbands,cbands) pair: transitions between
    those two sets are left out of the polarization, which is the
    constrained RPA of arXiv:0710.4013.

    kpoints/ek/ck let a caller that has already diagonalized on this mesh
    (a PairBasis, say) pass its own eigenstates in rather than repeating
    the work; they must be on the mesh get_kmesh(nk=nk) returns."""
    if h.has_eh:
        raise ValueError("the polarizability is not implemented for "
            "Nambu/BdG Hamiltonians (h.has_eh). The eigenstates of a BdG "
            "Hamiltonian are Bogoliubov quasiparticles, whose occupations "
            "are not fermion densities, so the density-density response "
            "built here would be meaningless -- and a gapped BdG spectrum "
            "is particle-hole symmetric about zero, so nothing downstream "
            "would notice. A superconducting screened interaction needs "
            "the Nambu-resolved response, not this one")
    if kpoints is None or ek is None or ck is None:
        kpoints,ek,ck = mesh_eigenstates(h,nk=nk)
    select_bands(ek) # validate: this raises if there is no gap at zero
    occ = np.array(ek<0.,dtype=np.float64) # Fermi energy is zero
    nb = ek.shape[1]
    allowed = np.ones((nb,nb),dtype=np.bool_)
    if exclude is not None: # constrained RPA
        vb,cb = exclude
        for n in vb:
            for m in cb: allowed[n,m] = False ; allowed[m,n] = False
        # Only occupancy-CHANGING pairs ever contribute, so the window
        # can be vacuous while `allowed` still has plenty of True entries
        # (its diagonal, for one). Check the transitions that matter.
        nocc = int(np.sum(occ[0]))
        left = [(n,m) for n in range(nb) for m in range(nb)
                if ((n<nocc)!=(m<nocc)) and allowed[n,m]]
        if len(left)==0:
            raise ValueError("the constrained-RPA window excludes every "
                "transition that could polarize the system, so chi0 is "
                "identically zero and W is just the bare interaction. "
                "This is what happens when the BSE band window is the "
                "whole spectrum, which is the default (nv=nc=None): "
                "there is then nothing 'outside' the window left to do "
                "the screening. Set nv/nc to a genuine subset of the "
                "bands, or use screening='rpa'")
    qs = kpoints # the q-mesh is the k-mesh
    ikq = mesh_sum_index(kpoints,qs)
    parallel.set_num_threads() # honor parallel.py's thread configuration
    chi0 = polarizability_jit(ck,ek,occ,ikq,allowed)
    chi0 = (chi0 + np.conj(np.transpose(chi0,(0,2,1))))/2. # exactly Hermitian
    return qs,chi0


def screened_interaction(h,V=None,nk=10,screening="rpa",exclude=None,
        kpoints=None,ek=None,ck=None,tol=1e-6):
    """Return the static RPA screened interaction of h as a
    ScreenedInteraction.

      V          the BARE interaction to screen, in the same convention
                 interaction.bare_interaction takes (None reads h.V).
                 Read the module docstring before passing a Hubbard U here
      nk         k-mesh for both the Brillouin zone sum and the q-grid W
                 is tabulated on
      screening  "rpa" (all transitions polarize) or "crpa" (transitions
                 inside the band window given by exclude do not)
      exclude    (vbands,cbands) for "crpa"; ignored for "rpa"

    Raises if the RPA diverges, i.e. if an eigenvalue of eps(q) reaches
    zero at some q -- that is a charge instability of the mean field at
    that wavevector, and the screened interaction there is infinite."""
    if screening not in ("rpa","crpa"):
        raise ValueError("screening must be 'rpa' or 'crpa', got %r"
                %(screening,))
    if screening=="rpa": exclude = None
    elif exclude is None:
        raise ValueError("screening='crpa' needs the band window to "
            "exclude from the polarization, as exclude=(vbands,cbands)")
    v = bare_interaction(h,V=V) # bare interaction, real-space dictionary
    qs,chi0 = static_polarizability(h,nk=nk,exclude=exclude,
            kpoints=kpoints,ek=ek,ck=ck)
    norb = chi0.shape[1]
    if norb!=h.intra.shape[0]:
        raise ValueError("the interaction and the Hamiltonian disagree on "
            "the number of orbitals (%d vs %d)"%(norb,h.intra.shape[0]))
    iden = np.identity(norb,dtype=np.complex128)
    Wq = np.zeros(chi0.shape,dtype=np.complex128)
    epsmin,absmin,qmin = None,None,None
    for iq,q in enumerate(qs):
        vq = interaction_at_q(v,h.geometry,q) # bare interaction at q
        eps = iden - vq@chi0[iq] # dielectric matrix
        ev = np.linalg.eigvals(eps)
        # Two conditions, because eps is a product of two Hermitian
        # matrices and neither is definite here, so its eigenvalues need
        # not be real. A real eigenvalue crossing zero is the physical
        # instability and shows up in the real part (and stays caught once
        # the system is past it, where the modulus would be large again);
        # a complex pair passing close to the origin is a near-singular
        # eps that the real part alone would miss.
        emin,amin = np.min(ev.real),np.min(np.abs(ev))
        if epsmin is None or emin<epsmin: epsmin,qmin = emin,q
        if absmin is None or amin<absmin: absmin = amin
        Wq[iq] = algebra.inv(eps)@vq
    if epsmin<=tol or absmin<=tol:
        raise ValueError("the RPA screening diverges: the dielectric "
            "matrix eps(q) = 1 - v(q) chi0(q) has an eigenvalue reaching "
            "zero (smallest real part %g, smallest modulus %g, tolerance "
            "%g; worst at q = %s), so W = eps^-1 v is infinite there. "
            "This is a charge or spin instability of the mean field at "
            "that wavevector (the same 1 - V chi = 0 condition "
            "chitk.rpa.rpa_kernel_poles reports as a collective mode), "
            "not a numerical problem: the reference state this is being "
            "built on is not stable against it. Reduce the interaction, "
            "or converge a mean field that already accounts for that "
            "order"%(epsmin,absmin,tol,list(qmin)))
    # W = v + v chi0 v + ... is Hermitian term by term; symmetrize away
    # the roundoff the inversion leaves behind
    Wq = (Wq + np.conj(np.transpose(Wq,(0,2,1))))/2.
    return ScreenedInteraction(qs,Wq,h.geometry,
            _nkmesh(h.geometry.dimensionality,nk),
            chi0=chi0,bare=v,epsmin=epsmin,screening=screening)


def _nkmesh(dim,nk):
    """Number of mesh points along each periodic direction, as get_kmesh
    lays them out: nk (or nk[i]) along the periodic directions, 1 along
    the rest. get_dict needs it to know how many lattice vectors the
    tabulated W(q) determines."""
    from ..checkclass import number2array
    out = np.ones(3,dtype=np.int64)
    nka = number2array(nk)
    for i in range(dim): out[i] = int(nka[i])
    return out
