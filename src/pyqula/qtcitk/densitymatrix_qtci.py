# qtci (quantics tensor cross interpolation) alternative to the density
# matrix of scftk/densitydensity.py's get_dm, for the self-consistent mean
# field (integration="qtci").
#
# This is the triangle integration of dmtk/simplexdm.py on a uniform
# 2^R x 2^R mesh in 2D (the tetrahedron one on a 2^R x 2^R x 2^R mesh in
# 3D, with three quantics coordinates) whose fields come from a cross interpolation instead of a
# diagonalization at every point. The density matrix at a lattice vector d,
#   dm[d]_ij = int dk sum_n f(E_n(k)) e^{2 pi i k.d} conj(u_n,i(k)) u_n,j(k),
# has a Fermi step that no smooth quadrature resolves, but it is the
# contraction of two smooth fields with the triangle occupation weights:
# the band energies E_n(k), and the field
#   X_n(k) = e^{2 pi i k.d} conj(u_n,i(k)) u_n,j(k),
# averaged over the degenerate multiplet of n at each k, which is the
# field simplexdm contracts. So the E_n, and the X_n of the entries the
# mean field reads, are reconstructed on the whole mesh from quantics TCIs
# (qtcitk/meshfields.py, shared with the qtci DOS and conductivity) that
# share one cache of diagonalizations, and the triangle weights of
# simplexdm, with the Bloechl term, are applied to the reconstructed mesh.
# Of the entries (d,i,j) and (-d,j,i), complex conjugates of each other,
# only one is interpolated.
#
# The fields are interpolated together, one TCI for the energies and one
# for all the X_n, with the field index as an extra tensor site
# (meshfields.qtci_stacked_fields), not one TCI per field. One TCI per
# field normalizes each field by its own maximum, so an entry that is zero
# up to rounding (the spin-off-diagonal entry of a collinear ferromagnet)
# is interpolated down to its noise at full rank: on the honeycomb
# ferromagnet of future_development/tetrahedron_density_matrix.md at
# 2^7 x 2^7 that took 49 s and every point of the mesh, against 1.4 s and
# 76% of the mesh for the stacked TCI, with the same error of 1e-6. The
# stacked TCI also shares its pivots in k between fields. The rank cap is
# meshfields.stacked_maxbonddim, 800, as in the qtci LDOS: on a layered
# four-band metal at 128^3 the X fields reached the cap of 400 of one
# field, with an error of 2.3e-5 against the tetrahedra, and 5e-9 at rank
# 512 under the cap of 800 (future_development/tetrahedron_density_matrix.md).
#
# The Fermi level is found on the same weights
# (simplexdm.fermi_from_energies), so the density matrix holds the
# requested charge to the accuracy of the interpolation, and the band
# energy is summed with the same weights. nk means the mesh: it is rounded
# up to a power of two, 2^R.
#
# The SCF loop finds the Fermi level of h and then asks for the density
# matrix of h shifted by that level. The shift moves every band energy by
# the same constant and leaves the eigenvectors alone, so the Fermi search
# keeps its eigenvectors and its reconstructed energies (_last_mesh), and
# the density matrix of the shifted Hamiltonian reuses both: one
# diagonalization per visited k-point serves the two passes of an
# iteration. The match is checked on the Bloch matrices themselves
# (_shift_from), so any other Hamiltonian starts from scratch.
#
# The bands are ordered by energy at each k, so two bands that cross give
# fields with a kink, which costs rank, as in the qtci DOS.
#
# The earlier qtci density matrix integrated each entry with a tensorized
# Gauss-Kronrod rule of 4 ceil(log2 nk)+1 nodes per axis. That is spectral
# in a gap, but it does not resolve a metallic Fermi step at any nk, and it
# was removed (future_development/tetrahedron_density_matrix.md has the
# measurements, including the comparison on a gapped state).
import numpy as np

from .. import parallel
from ..kpmtk.densitymatrix_kpm import required_elements, required_elements_eh
from .meshfields import qtci_stacked_fields, stacked_maxbonddim

DEFAULT_NK = 8


def mesh_exponent(nk):
    """R of the 2^R mesh for a requested nk, rounded up"""
    if nk is None: nk = DEFAULT_NK
    return max(1,int(np.ceil(np.log2(nk))))


def check_hamiltonian(h,what):
    """Refuse what the qtci density matrix does not cover"""
    if h.dimensionality not in (2,3):
        raise NotImplementedError(what+" is only implemented in 2D and 3D "
                "(the quantics interpolation runs on a 2^R x 2^R or "
                "2^R x 2^R x 2^R mesh), got dimensionality %d; use "
                "integration=\"triangle\""%h.dimensionality)
    if getattr(h,"non_hermitian",False):
        raise NotImplementedError(what+" needs a Hermitian Hamiltonian; a "
                "non-Hermitian one is solved with its biorthogonal density "
                "matrix, which only integration=\"ed\" builds")


class _Eigen():
    """Eigenpairs of h at the points of the 2^R mesh (2^R x 2^R in 2D,
    2^R x 2^R x 2^R in 3D), each point diagonalized once; ndiag counts the
    diagonalizations"""
    def __init__(self,h,R):
        if h.is_sparse: h = h.get_dense()
        self.hkgen = h.get_hk_gen()
        self.R = R
        self.dim = h.dimensionality
        self.n = 2**R
        self.nb = h.intra.shape[0]
        self.es = {} # (i,j) -> energies
        self.U = {} # (i,j) -> eigenvectors as columns
        self.ndiag = 0
    def _diagonalize(self,ks):
        from ..dostk.simplexdos import _hk_batch
        from ..htk.eigenvectors import parallel_diagonalization
        parallel.set_num_threads()
        es,U = parallel_diagonalization(_hk_batch(self.hkgen,ks))
        self.ndiag += len(ks)
        return es,U
    def __call__(self,ks):
        """energies (m,nb) and eigenvectors (m,nb,nb) at the mesh points ks"""
        ij = [tuple(q) for q in np.rint(np.asarray(ks)[:,:self.dim]*self.n
            ).astype(np.int64).tolist()]
        new = [q for q in dict.fromkeys(ij) if q not in self.es]
        if new:
            knew = np.zeros((len(new),3))
            knew[:,:self.dim] = np.array(new,dtype=np.float64)/self.n
            es,U = self._diagonalize(knew)
            for q,e,u in zip(new,es,U):
                self.es[q] = e
                self.U[q] = u
        return (np.array([self.es[q] for q in ij]),
                np.array([self.U[q] for q in ij]))


def _bloch_data(h):
    if h.is_sparse: return None
    return getattr(h.get_hk_gen(),"bloch_data",None)


_last_mesh = None # what the last Fermi search reconstructed, see _shift_from


def _shift_from(h,last,R,tolerance):
    """c when h is the Hamiltonian of the last Fermi search plus c times the
    identity, on the same mesh and at the same tolerance; None otherwise"""
    if last is None or last["R"]!=R or last["tolerance"]!=tolerance:
        return None
    data = _bloch_data(h)
    if data is None: return None
    ms,ds = data
    ms0,ds0 = last["bloch"]
    if ms.shape!=ms0.shape or not np.array_equal(ds,ds0): return None
    diff = ms-ms0
    onsite = np.all(ds==0.,axis=1)
    if np.any(diff[~onsite]!=0.): return None
    d0 = np.sum(diff[onsite],axis=0)
    c = d0[0,0].real
    scale = 1e-12*max(1.,np.max(np.abs(ms0)))
    if np.max(np.abs(d0-c*np.identity(len(d0))))>scale: return None
    return c


def _energies(h,R,tolerance):
    """band energies of h reconstructed on the 2^R mesh, with the
    eigenpair cache they were computed with, the number of
    diagonalizations this call did and the bond dimensions of the TCI.
    When h is the Hamiltonian of the last Fermi search shifted by a
    constant, that search's energies and eigenpairs are reused"""
    c = _shift_from(h,_last_mesh,R,tolerance)
    if c is not None:
        m = _last_mesh
        return m["es"]+c,m["eigen"],0,m["ranks"]
    eigen = _Eigen(h,R)
    nb = eigen.nb
    es,ncalls,ranks = qtci_stacked_fields(lambda ks: eigen(ks)[0],R,nb,
            tolerance=tolerance,dim=h.dimensionality,
            maxbonddim=stacked_maxbonddim)
    return np.ascontiguousarray(es),eigen,eigen.ndiag,ranks


def get_fermi4filling_qtci(h,filling,nk=None,T=1e-7,tolerance=1e-6):
    """Fermi energy that holds `filling` (a fraction of the states) under
    the qtci density matrix: the triangle charge of the band energies
    reconstructed on the 2^R mesh, which is continuous in the Fermi
    level, so the filling is reached exactly on that mesh. The energies
    and the eigenpairs are kept, so that the density matrix of h shifted
    by this level does not diagonalize the same points again. For a
    Bogoliubov-de Gennes Hamiltonian the count is done on the normal
    state, the same approximation spectrum.get_fermi4filling makes; a mean
    field corrects it on the charge of the paired state
    (scftk.superscf.BdGFermiLevel)"""
    global _last_mesh
    if h.has_eh:
        h0 = h.copy()
        h0.remove_nambu()
        return get_fermi4filling_qtci(h0,filling,nk=nk,T=T,
                tolerance=tolerance)
    check_hamiltonian(h,"the qtci Fermi level")
    from ..dmtk import simplexdm
    R = mesh_exponent(nk)
    es,eigen,nd,ranks = _energies(h,R,tolerance)
    mu = simplexdm.fermi_from_energies(h.dimensionality,2**R,es,filling,T=T)
    data = _bloch_data(h)
    # the eigenvectors are kept for the next call only below the memory
    # limit of the triangle mesh; above it the density matrix starts over
    keep = len(eigen.U)*eigen.nb**2*16 <= simplexdm.max_memory_gb*1e9
    if data is not None and keep:
        _last_mesh = dict(R=R,tolerance=tolerance,es=es,eigen=eigen,
                ranks=ranks,bloch=(data[0].copy(),data[1].copy()))
    return mu


def _canonical_entries(needed):
    """the entries to interpolate: of (d,i,j) and (-d,j,i), whose fields
    are complex conjugates of each other, only one"""
    out = set()
    for (d,i,j) in needed:
        d = tuple(int(x) for x in d)
        other = (tuple(-x for x in d),j,i)
        out.add(min((d,i,j),other))
    return sorted(out)


def dm_entries_qtci(h,needed,nk=None,T=1e-7,tolerance=1e-6,bloechl=None,
        counted=False):
    """The density-matrix entries `needed`, a set of (d,i,j), of h at zero
    chemical potential, as {d: (n,n) matrix} with the other entries zero.
    Returns also the number of diagonalizations and the bond dimensions of
    the cross interpolations when counted"""
    check_hamiltonian(h,"the qtci density matrix")
    from ..dmtk import simplexdm
    from ..dostk.simplexdos import _multiplet_average, _degeneracy_tol
    if bloechl is None: bloechl = simplexdm.bloechl_default
    R = mesh_exponent(nk)
    n = 2**R
    dim = h.dimensionality
    es,eigen,ndiag,eranks = _energies(h,R,tolerance)
    nd0 = eigen.ndiag
    entries = _canonical_entries(needed)
    nb = eigen.nb
    norb = h.intra.shape[0]
    dvecs = np.array([e[0] for e in entries],dtype=np.float64).reshape(-1,3)
    I = np.array([e[1] for e in entries],dtype=np.int64)
    J = np.array([e[2] for e in entries],dtype=np.int64)
    ne = len(entries)
    def evaluate(ks):
        e,U = eigen(ks)
        W = np.transpose(U[:,I,:].conj()*U[:,J,:],(0,2,1)) # (k,band,entry)
        W = np.ascontiguousarray(W)
        _multiplet_average(e,W,_degeneracy_tol(e))
        ph = np.exp(2j*np.pi*(np.asarray(ks)@dvecs.T)) # (k,entry)
        return (W*ph[:,None,:]).reshape(len(ks),nb*ne)
    ds = sorted({tuple(int(x) for x in d) for (d,i,j) in needed}
            |{tuple(int(x) for x in e[0]) for e in entries})
    dm = {d: np.zeros((norb,norb),dtype=np.complex128) for d in ds}
    xranks = []
    if ne>0:
        X,ncalls,xranks = qtci_stacked_fields(evaluate,R,nb*ne,
                tolerance=tolerance,dtype=np.complex128,dim=dim,
                maxbonddim=stacked_maxbonddim)
        w = simplexdm.simplex_weights(dim,n,es,mu=0.,T=T,bloechl=bloechl)
        vals = np.einsum("kn,kne->e",w,X.reshape(n**dim,nb,ne),optimize=True)
        for (d,i,j),x in zip(entries,vals):
            dm[d][i,j] = x
            dm[tuple(-y for y in d)][j,i] = np.conj(x)
    # only the entries asked for, as the other backends that compute a
    # subset leave the rest at zero
    out = {d: np.zeros((norb,norb),dtype=np.complex128) for d in ds}
    for (d,i,j) in needed:
        d = tuple(int(x) for x in d)
        out[d][i,j] = dm[d][i,j]
    if counted: return out,ndiag+eigen.ndiag-nd0,list(eranks)+list(xranks)
    return out


def get_dm_qtci(h,v,nk=None,fermi=0.0,T=1e-7,tolerance=1e-6,bloechl=None,
        counted=False):
    """qtci analogue of scftk.densitydensity.get_dm: a dictionary
    {direction: matrix} with the density matrix of h at zero chemical
    potential (fermi shifts it), computing only the entries the interaction
    v requires, from the simplex weights on the 2^R mesh (nk rounded
    up to a power of two) with the band energies and the band projectors
    reconstructed from quantics cross interpolations (see the module
    comment). So this is integration="triangle" at the same mesh, up to
    the interpolation tolerance, with fewer diagonalizations when the
    fields have a low rank. It works for a metal as for an insulator.

    The interpolation adds an error to each entry, 8e-9 against the
    triangle on the honeycomb ferromagnet at 64x64 and tolerance=1e-6
    (3e-10 at 1e-8), and the pivots of the cross interpolation change from
    one iteration to the next. So the SCF residual has a floor of that
    size, and maxerror has to stay above it: that mean field did not reach
    maxerror=1e-9 in 400 iterations at the default tolerance, and stops
    at 1e-7 as the triangle does at 1e-9 (future_development/
    tetrahedron_density_matrix.md), as for a KPM mean field.

    A direction of v with no required entry gets an all-zero matrix, a
    placeholder that is correct for the mean field, as in
    kpmtk.densitymatrix_kpm.get_dm_kpm. For a Bogoliubov-de Gennes
    Hamiltonian the entries are those of required_elements_eh, and the
    weights are those of the quasiparticle bands at zero energy.
    bloechl=None takes dmtk.simplexdm.bloechl_default. With counted=True
    also returns the number of diagonalizations and the bond dimensions of
    the cross interpolations"""
    if h.has_eh:
        # and the diagonal of the onsite matrix, whose electron block is
        # the charge a Bogoliubov-de Gennes mean field holds at a filling
        # (scftk.superscf.get_charge_bdg), also on sites with no interaction
        needed = set(required_elements_eh(v))
        needed |= {((0,0,0),i,i) for i in range(h.intra.shape[0])}
    else: needed = required_elements(v)
    if fermi!=0.:
        h = h.copy()
        h.shift_fermi(-fermi)
    res = dm_entries_qtci(h,needed,nk=nk,T=T,tolerance=tolerance,
            bloechl=bloechl,counted=counted)
    dm = res[0] if counted else res
    norb = h.intra.shape[0]
    for d in [(0,0,0)]+[tuple(d) for d in v]:
        if d not in dm: dm[d] = np.zeros((norb,norb),dtype=np.complex128)
    if counted: return (dm,)+tuple(res[1:])
    return dm


def full_dm_qtci(h,ds,nk=None,T=1e-7,tolerance=1e-6,bloechl=None):
    """Every entry of the density matrix for the lattice vectors ds, with
    the construction of get_dm_qtci. The SCF loop only computes the
    entries the mean field reads; this is the full matrix, for reporting
    it once the loop is done"""
    norb = h.intra.shape[0]
    needed = {(tuple(int(x) for x in d),i,j) for d in ds
            for i in range(norb) for j in range(norb)}
    return dm_entries_qtci(h,needed,nk=nk,T=T,tolerance=tolerance,
            bloechl=bloechl)


def band_energy_qtci(h,nk=None,tolerance=1e-6,bloechl=None):
    """Sum of the band energies below zero per unit cell, with the triangle
    weights at T=0 (and the Bloechl term, bloechl=None takes
    dmtk.simplexdm.bloechl_default) on the band energies reconstructed on
    the 2^R mesh, as simplexdm.band_energy does on the exact ones.
    For a Bogoliubov-de Gennes Hamiltonian it is the electronic energy,
    E = (sum_{E<0} E_BdG + Tr h_e)/2, with Tr h_e the onsite trace of the
    electron block, which is the mean of Tr h_e(k) over the zone"""
    check_hamiltonian(h,"the qtci band energy")
    from ..dmtk import simplexdm
    if bloechl is None: bloechl = simplexdm.bloechl_default
    R = mesh_exponent(nk)
    es,eigen,nd,ranks = _energies(h,R,tolerance)
    e = simplexdm.band_energy_from_energies(h.dimensionality,2**R,es,
            bloechl=bloechl)
    if h.has_eh:
        from .. import operators, algebra
        pe = np.array(algebra.todense(operators.get_electron(h)))
        intra = np.array(algebra.todense(h.intra))
        e = (e+np.sum(pe.diagonal().real*intra.diagonal().real))/2.
    return e
