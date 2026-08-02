import numpy as np
from .. import filesystem as fs
from ..dos import write_dos
from ..ldos import write_ldos
from .impurity import build_impurity_hamiltonian
from .ldosmap import real_space_ldos
from .fourier import ldos_fourier_transform,commensurate_qmesh


def get_qpi_impurity(h,nsuper=10,impurities=[],energies=0.0,
        delta=0.05,num_waves=20,nk=2,
        write=True,output_folder="QPI_IMPURITY",**kwargs):
    """Real-space-impurity QPI: build a supercell of h, add a handful of
    real-space impurities (see qpitk.impurity.build_impurity_hamiltonian
    for the impurity spec format), compute the real-space LDOS map with
    ARPACK partial diagonalization (num_waves eigenstates around each
    energy, see qpitk.ldosmap.real_space_ldos -- efficient for large
    supercells, unlike full diagonalization), and Fourier transform it
    directly (a discrete sum over the actual atomic positions, not a
    grid FFT, see qpitk.fourier.ldos_fourier_transform) to get the
    QPI(q) signal, evaluated at the q commensurate with the supercell
    (qpitk.fourier.commensurate_qmesh).

    No band unfolding is needed: this never diagonalizes supercell
    bands and projects them onto primitive Bloch states (which is what
    unfolding, e.g. get_qpi's nunfold/store_primal, is for) -- it only
    Fourier transforms a real-space scalar density, evaluated directly
    at q spanning the full primitive Brillouin zone. Because the
    impurity pattern repeats with the supercell period, the achievable
    q *resolution* (not the BZ range) is what nsuper sets: exactly
    nsuper1*nsuper2 independent q values are available, fixed by the
    commensurate mesh, not by a separate free plotting resolution.

    num_waves is a *starting point*, not a hard cap: real_space_ldos
    grows it as needed (see qpitk.ldosmap._eigenstates_for_energy_window)
    so the diagonalization always both reaches margin*delta past every
    requested energy and never stops in the middle of a degenerate
    manifold -- picking it too small just costs extra ARPACK calls to
    grow from, not correctness. Pass margin= (default 5.0) to control
    how far past each energy that coverage check reaches.

    Returns (r,ldos_r,q,qpi_q):
      r: (nsites,3) real-space positions of the supercell
      ldos_r: real-space LDOS, (nsites,) for a scalar energy or
        (nenergies,nsites) otherwise
      q: (nsuper1*nsuper2,3) q-mesh spanning the primitive BZ
      qpi_q: QPI intensity |FT(ldos_r)|, (nsuper1*nsuper2,) or
        (nenergies,nsuper1*nsuper2), matching ldos_r's shape"""
    if h.dimensionality!=2: raise ValueError("get_qpi_impurity is only implemented for 2D Hamiltonians")
    g0 = h.geometry # primitive geometry, needed for the commensurate q-mesh
    hs = build_impurity_hamiltonian(h,nsuper,impurities)
    scalar_energy = np.ndim(energies)==0
    energies_arr = np.atleast_1d(np.array(energies,dtype=np.float64))
    r,ldos_r = real_space_ldos(hs,energies_arr,delta=delta,
            num_waves=num_waves,nk=nk,**kwargs)
    q = commensurate_qmesh(g0,nsuper)
    qpi_q = np.array([np.abs(ldos_fourier_transform(r,d,q)) for d in ldos_r])
    if write: _write_qpi_impurity(output_folder,r,ldos_r,q,qpi_q,energies_arr)
    if scalar_energy: return r,ldos_r[0],q,qpi_q[0]
    return r,ldos_r,q,qpi_q


def _write_qpi_impurity(output_folder,r,ldos_r,q,qpi_q,energies):
    """Write one LDOS and one QPI file per energy, plus a combined DOS
    file, mirroring the MULTIQPI/ folder convention used by get_qpi.
    Reuses the existing dos.write_dos/ldos.write_ldos writers for the
    DOS/LDOS files (same format, avoids the two on-disk conventions
    silently drifting apart); the QPI(q) file has no analogous existing
    writer (get_qpi's own mode="pm"/"response" paths hand-roll it with
    np.savetxt too, see chitk/qpi.py), so this does the same here."""
    fs.rmdir(output_folder)
    fs.mkdir(output_folder)
    dos = np.sum(ldos_r,axis=1)
    write_dos(energies,dos,output_file=output_folder+"/DOS.OUT")
    fo = open(output_folder+"/"+output_folder+".TXT","w")
    for ie,e in enumerate(energies):
        ldos_name = "LDOS_"+str(e)+"_.OUT"
        write_ldos(r[:,0],r[:,1],ldos_r[ie],output_file=output_folder+"/"+ldos_name)
        qpi_name = "QPI_"+str(e)+"_.OUT"
        np.savetxt(output_folder+"/"+qpi_name,
                np.array([q[:,0],q[:,1],qpi_q[ie]]).T)
        fo.write(qpi_name+"\n")
    fo.close()
