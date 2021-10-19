
import numpy as np
from copy import deepcopy
from ..superconductivity import get_eh_sector
from ..superconductivity import build_nambu_matrix



# different constrains for the meanfield calculation

def remove_onsite_spinful(m):
    n = m.shape[0]//2 # number of orbitals
    for i in range(n):
        d = (m[2*i,2*i] + m[2*i+1,2*i+1])/2. # average charge
        m[2*i,2*i] -= d # set
        m[2*i+1,2*i+1] -= d # set
    return m



def remove_magnetism_spinful(m):
    n = m.shape[0]//2 # number of orbitals
    for i in range(n):
        d = (m[2*i,2*i] + m[2*i+1,2*i+1])/2. # average charge
        m[2*i,2*i] = d # set
        m[2*i+1,2*i+1] = d # set
        m[2*i,2*i+1] = 0. # set
        m[2*i+1,2*i] = 0. # set
    return m


def remove_offplane_magnetism_spinful(m):
    n = m.shape[0]//2 # number of orbitals
    for i in range(n):
        d = (m[2*i,2*i] + m[2*i+1,2*i+1])/2. # average charge
        m[2*i,2*i] = d # set
        m[2*i+1,2*i+1] = d # set
    return m



def remove_inplane_magnetism_spinful(m):
    n = m.shape[0]//2 # number of orbitals
    for i in range(n):
        m[2*i,2*i+1] = 0. # set
        m[2*i+1,2*i] = 0. # set
    return m




def remove_spinful_sector(h,removef):
    """Remove total charge renormalization"""
    has_eh = h.has_eh
    has_spin = h.has_spin
    def f(dd): # create function
        out = deepcopy(dd) # copy the dictionary
        m = out[(0,0,0)] # onsite matrix
        if has_eh and not has_spin: raise # not implemented
        elif not has_eh and has_spin: # spinful
            m = removef(m)
        elif has_eh and has_spin: # spinful
            m01 = get_eh_sector(m,i=0,j=1) # anomalous part
            m10 = get_eh_sector(m,i=1,j=0) # anomalous part
            m00 = get_eh_sector(m,i=0,j=0) # anomalous part
            m00 = removef(m00) # remove onsite 
            m = build_nambu_matrix(m00,c12=m01,c21=m10) # rebuild the matrix
        else: raise
        out[(0,0,0)] = m # set the new matrix
        return out # return dictionary
    return f # return function


def remove_charge(h):
    return remove_spinful_sector(h,remove_onsite_spinful)

def remove_magnetism(h):
    return remove_spinful_sector(h,remove_magnetism_spinful)

def remove_inplane_magnetism(h):
    return remove_spinful_sector(h,remove_inplane_magnetism_spinful)

def remove_offplane_magnetism(h):
    return remove_spinful_sector(h,remove_offplane_magnetism_spinful)




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
            else: raise
        elif c in ["no_anomalous_term","no_SC","no_superconductivity"]:
            if h.has_eh:
                from ..sctk.extract import extract_normal_dict
                mf = extract_normal_dict(mf)
            else: pass
#        print(np.round(mf[(0,0,0)],1))
    return mf




def obj2mf(mf):
    from ..algebra import ismatrix
    from ..hamiltonians import Hamiltonian
    if ismatrix(mf): return {(0,0,0):mf}
    elif type(mf)==Hamiltonian: return mf.get_dict()
    else: return mf # assume it is a valid dictionary

