
import numpy as np
from copy import deepcopy
from scipy.sparse import issparse,csc_matrix
from ..superconductivity import get_eh_sector
from ..superconductivity import build_nambu_matrix



# different constrains for the meanfield calculation

def _get_entries(m,rows,cols):
    """The entries (rows[k],cols[k]) of a dense or sparse matrix"""
    return np.asarray(m[rows,cols]).ravel()


def _set_entries(m,rows,cols,values):
    """Return m with its entries (rows[k],cols[k]) set to values, all in
    one step. Setting them one at a time on a sparse matrix rebuilds it
    every time its pattern changes, which made these constraints
    quadratic in the number of sites"""
    if issparse(m):
        delta = values - _get_entries(m,rows,cols)
        delta = csc_matrix((delta,(rows,cols)),shape=m.shape)
        return (m + delta).asformat(m.format)
    m[rows,cols] = values
    return m


def _spin_indexes(m):
    """Indexes of the up and down orbital of every site"""
    i = np.arange(m.shape[0]//2) # number of orbitals
    return 2*i,2*i+1


def remove_onsite_spinful(m):
    up,dn = _spin_indexes(m)
    a,b = _get_entries(m,up,up),_get_entries(m,dn,dn)
    d = (a + b)/2. # average charge
    return _set_entries(m,np.concatenate([up,dn]),np.concatenate([up,dn]),
            np.concatenate([a-d,b-d]))


def remove_onsite_spinless(m):
    i = np.arange(m.shape[0]) # number of orbitals
    return _set_entries(m,i,i,np.zeros(len(i)))


def remove_magnetism_spinful(m):
    up,dn = _spin_indexes(m)
    d = (_get_entries(m,up,up) + _get_entries(m,dn,dn))/2. # average charge
    z = np.zeros(len(up))
    return _set_entries(m,np.concatenate([up,dn,up,dn]),
            np.concatenate([up,dn,dn,up]),np.concatenate([d,d,z,z]))


def remove_offplane_magnetism_spinful(m):
    up,dn = _spin_indexes(m)
    d = (_get_entries(m,up,up) + _get_entries(m,dn,dn))/2. # average charge
    return _set_entries(m,np.concatenate([up,dn]),np.concatenate([up,dn]),
            np.concatenate([d,d]))


def remove_inplane_magnetism_spinful(m):
    up,dn = _spin_indexes(m)
    z = np.zeros(len(up))
    return _set_entries(m,np.concatenate([up,dn]),np.concatenate([dn,up]),
            np.concatenate([z,z]))


def remove_spinless_sector(h,removef,alldirs=True):
    """Remove total charge renormalization.

    alldirs=True applies removef to EVERY direction of the mean-field
    dictionary, not just the onsite (0,0,0) block: with an intersite
    interaction the spin-dependent Fock term lives on the bonds, so a
    constraint that only rewrote (0,0,0) was a silent no-op there. It is
    False for the charge constraint, whose remover (remove_onsite_*) is
    about the onsite charge specifically -- applied to a bond it would
    delete the spin-independent part of the hopping renormalization (a
    bond/Kekule-type charge order), which is not what "no_charge" says."""
    has_eh = h.has_eh
    has_spin = h.has_spin
    if has_eh:
        raise NotImplementedError("remove_spinless_sector is not implemented "
                "for Hamiltonians with the electron-hole (Nambu) degree of "
                "freedom")
    if has_spin: # not the right function
        raise ValueError("remove_spinless_sector is for spinless "
                "Hamiltonians, use the spinful version instead")
    def f(dd): # create function
        out = deepcopy(dd) # copy the dictionary
        for d in (out if alldirs else [(0,0,0)]): # loop over directions
            out[d] = removef(out[d]) # remove the sector
        return out # return dictionary
    return f # return function




def remove_spinful_sector(h,removef,alldirs=True):
    """Remove total charge renormalization.

    alldirs: see remove_spinless_sector's docstring -- every direction of
    the mean field, or only the onsite (0,0,0) block. The spin structure
    of a bond matrix is the same 2x2 block per pair of orbitals that the
    onsite one has, so the removers apply unchanged; and for a Nambu
    Hamiltonian the hole block is rebuilt from the (constrained) electron
    block at the SAME direction, which is the relation
    superconductivity.build_nambu_matrix already keeps at every
    direction (the hole Hamiltonian is -h*(-k), whose real-space hopping
    at d is -conj(t_d), not -conj(t_-d))."""
    has_eh = h.has_eh
    has_spin = h.has_spin
    def f(dd): # create function
        out = deepcopy(dd) # copy the dictionary
        for d in (out if alldirs else [(0,0,0)]): # loop over directions
            m = out[d] # matrix of this direction
            if not has_eh and has_spin: # spinful
                m = removef(m)
            elif has_eh and has_spin: # spinful
                m01 = get_eh_sector(m,i=0,j=1) # anomalous part
                m10 = get_eh_sector(m,i=1,j=0) # anomalous part
                m00 = get_eh_sector(m,i=0,j=0) # anomalous part
                m00 = removef(m00) # remove the sector
                m = build_nambu_matrix(m00,c12=m01,c21=m10) # rebuild the matrix
            else:
                raise NotImplementedError("this Hilbert space is not "
                        "implemented in the mean-field constrain")
            out[d] = m # set the new matrix
        return out # return dictionary
    return f # return function


def remove_charge(h):
    # alldirs=False: the onsite charge only, see remove_spinless_sector
    if h.has_spin:
        return remove_spinful_sector(h,remove_onsite_spinful,alldirs=False)
    else:
        return remove_spinless_sector(h,remove_onsite_spinless,alldirs=False)

def remove_magnetism(h):
    return remove_spinful_sector(h,remove_magnetism_spinful)

def remove_inplane_magnetism(h):
    return remove_spinful_sector(h,remove_inplane_magnetism_spinful)

def remove_offplane_magnetism(h):
    return remove_spinful_sector(h,remove_offplane_magnetism_spinful)




known_constrains = ["no_charge","no_magnetism","no_inplane_magnetism",
        "no_offplane_magnetism","no_normal_term","no_anomalous_term",
        "no_SC","no_superconductivity"]


def enforce_constrains(mf,h,constrains=[]):
    """Given a list of constrains, return a function that enforces
    all of them in the mean field"""
    for c in constrains:
        if c=="no_charge":
            mf = remove_charge(h)(mf) # remove charge renormalization
        elif c=="no_magnetism":
            mf = remove_magnetism(h)(mf) # remove magnetism
        elif c=="no_inplane_magnetism":
            mf = remove_inplane_magnetism(h)(mf) # remove inplane magnetism
        elif c=="no_offplane_magnetism":
            mf = remove_offplane_magnetism(h)(mf) # remove inplane magnetism
        elif c=="no_normal_term":
            if h.has_eh:
                from ..sctk.extract import extract_anomalous_dict
                mf = extract_anomalous_dict(mf)
            else:
                raise ValueError("constrain 'no_normal_term' only makes "
                  +"sense for a Hamiltonian with the electron-hole (Nambu) "
                  +"degree of freedom; this one has has_eh=False")
        elif c in ["no_anomalous_term","no_SC","no_superconductivity"]:
            if h.has_eh:
                from ..sctk.extract import extract_normal_dict
                mf = extract_normal_dict(mf)
            else: pass # no anomalous term to remove in the first place
        else:
            raise ValueError("unknown mean-field constrain '"+str(c)
              +"'. Known constrains: "+str(known_constrains))
#        print(np.round(mf[(0,0,0)],1))
    return mf




def obj2mf(mf):
    from ..algebra import ismatrix
    from ..hamiltonians import Hamiltonian
    if ismatrix(mf): return {(0,0,0):mf}
    elif isinstance(mf, Hamiltonian): return mf.get_dict()
    else: return mf # assume it is a valid dictionary

