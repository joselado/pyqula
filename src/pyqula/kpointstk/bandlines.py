from .labels import label2k
from .locate import same_kpoint
import numpy as np

def write_bandlines(g,ks,labels):
    """Given a set of kpoints and labels, write the bandlines in a file"""
    f = open("BANDLINES.OUT","w")
    KHS = [label2k(g,l) for l in labels] # high symmetry points
    def check(k):
        for j in range(len(labels)):
            if same_kpoint(k,KHS[j]): return labels[j] # same kpoint
        return None
    f = open("BANDLINES.OUT","w")
    for i in range(len(ks)): # loop over kpoints
        c = check(ks[i]) # check
        if c is not None: # check
            f.write(str(i)+"  "+str(c)+"\n")
    f.close() # close file







def write_bandlines_indexes(iks,labels):
    """Write a file with the special kpoints"""
    f = open("BANDLINES.OUT","w")
    for (ik,l) in zip(iks,labels):
        f.write(str(ik)+"  "+str(l)+"\n")
    f.close()



