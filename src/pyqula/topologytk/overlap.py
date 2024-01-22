import numpy as np


def uij(wf1,wf2):
  """ Calcultes the matrix product of two sets of input wavefunctions"""
  out =  np.matrix(np.conjugate(wf1))@(np.matrix(wf2).T)  # faster way
  return out


def uij_slow(wf1,wf2):
  m = np.matrix(np.zeros((len(wf1),len(wf2)),dtype=np.complex_))
  for i in range(len(wf1)):
    for j in range(len(wf2)):
      m[i,j] = np.conjugate(wf1[i]).dot(wf2[j])
  return m


