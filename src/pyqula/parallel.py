# routines to call a function in parallel
import scipy.linalg as lg
from . import algebra
import numba

numba_cores = None # use all cores

def set_num_threads():
    from .paralleltk.multiprocess import _is_worker
    if not _is_worker: # main thread
        if numba_cores is not None:
            numba.set_num_threads(numba_cores) # set these cores
    else:
        numba.set_num_threads(1) # single thread

#from .paralleltk.multiprocess import set_cores
#from .paralleltk.multiprocess import pcall
#from .paralleltk.multiprocess import _num_cores as cores
#from .paralleltk.jlib import pcall
#from .paralleltk.jlib import set_cores
#from .paralleltk.jlib import _cores as cores

# parallelization not working yet, workaround
cores = 1
def pcall(f,xs,**kwargs):  return [f(x) for x in xs]
def set_cores(n):  
    global cores
    cores=1



