import numpy as np
import os
from scipy.sparse import csr_matrix
import pickle
from . import filesystem as fs

def save_sparse_csr(filename,array):
    array = csr_matrix(array)
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename,is_sparse=True):
    loader = np.load(filename)
    m = csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
    if not is_sparse: m = m.todense() # dense matrix
    return m



def write(x,y,output_file="DATA.OUT",comment=None):
  fo = open(output_file,"w")
  if comment is not None: fo.write("# "+comment+"\n")
  for (ix,iy) in zip(x,y):
    fo.write(str(ix)+"    "+str(iy)+"\n")
  fo.close()



def save_sparse_pairs(filename,pairs):
  """Saves pairs of tuple and sparse matrix in a folder"""
  fs.rmdir(filename) # remove folder
  fs.mkdir(filename) # create folder
  fo = open(filename+"/files.txt","w") # names of the files
  for (p,m) in pairs: # loop over the pairs
    name = ""
    for ip in p: name += str(ip)+"_" # create a name
    name += ".npz"
    save_sparse_csr(filename+"/"+name,csr_matrix(m)) # save matrix
    fo.write(name+"\n") # save name of the file
  fo.close()


def read_sparse_pairs(filename,is_sparse=True):
  """Saves pairs of tuple and sparse matrix in a folder"""
  names = open(filename+"/files.txt").readlines() # names of the files
  names = [name.replace("\n","") for name in names] # remove \n
  pairs = [] # empty list
  for name in names: # loop over the pairs
    m = load_sparse_csr(filename+"/"+name) # read matrix
    if not is_sparse: m = m.todense() # dense matrix
    v = name.split("_") # slpit in numbers
    v = np.array([int(v[i]) for i in range(len(v)-1)]) # set as array
    pairs.append((v,m)) # append this pair
  return pairs





def load(input_file):
    """Load the hamiltonian"""
    with open(input_file, 'rb') as input:
      return pickle.load(input)

def save(self,output_file):
  """ Write an object"""
  with open(output_file, 'wb') as output:
    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)




def writefile(original,write=True,filename="FILE.OUT"):
  def wrapper(target):
      out = target(*args,**kwargs)
      if write: np.savetxt(filename,np.array(out).T)
      return out
  return wrapper

