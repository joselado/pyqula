import numpy as np
from . import algebra

def get_waves(m,non_hermitian=False,**kwargs):
    """Return the eigenenergies and eigenwaves"""
    if not non_hermitian: 
        return get_waves_hermitian(m,**kwargs)
    else:
        return get_waves_non_hermitian(m,**kwargs)




def get_waves_hermitian(intra,num_bands=None,**kwargs):
    """Return eigenvalues and eigenvectors"""
    if num_bands is None:
        eig,eigvec = algebra.eigh(intra)
        eigvec = eigvec.T # transpose
    else:
        eig,eigvec = algebra.smalleig(intra,numw=num_bands,evecs=True)
    return eig,eigvec



def get_waves_non_hermitian(intra,num_bands=None,eigmode="complex",
        **kwargs):
    """Return eigenvalues and eigenvectors"""
    eig,eigvec = algebra.eig(intra)
    eigvec = eigvec.T # transpose
    if eigmode=="complex": pass
    elif eigmode=="real": eig = eig.real
    elif eigmode=="imag": eig = eig.imag
    else: raise
    return eig,eigvec

