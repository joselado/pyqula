# routines to call a function in parallel
from __future__ import print_function
import scipy.linalg as lg
from . import algebra


is_child = False # check if you are running a child

try:
  from multiprocess import Pool
  import multiprocess
  maxcpu = multiprocess.cpu_count()
except:
    print("Multiprocess not working")
    def Pool(n=1): # workaround
            class mpool():
                def map(self,f,xs):
                  return [f(x) for x in xs]
                def terminate(self): return None # dummy function
            return mpool()
    maxcpu = 1


cores = 1 # call in a single by default


def set_cores(n=1):
    global cores
    cores = n


#mainpool = None

#def initialize(): 
#  global mainpool
#  if cores>1:
#    mainpool = Pool(cores) # create pool
#  return mainpool

#def finish(): mainpool=None # delete pool


def multieigh(ms):
  """Diagonalize a bunch of Hamiltonians at once"""
#  mainpool = initialize()
  if mainpool is not None: mainpool.map(lg.eigh,ms)
  else: return [algebra.eigh(m) for m in ms]





def pcall_serial(fun,args):
  """Function to call in serial"""
  return [fun(a) for a in args]


#def pcall_mp(fun,args,cores=1): return pcall_serial(fun,args)

#try: # try to use the multiprocessing library
#  from pathos.multiprocessing import Pool
def pcall_mp(fun,args,cores=cores):
    """Calls a function for every input in args"""
    mainpool = Pool(cores) # create pool
#    print("Using",cores,"cores")
    out = mainpool.map(fun,args) # return list
    mainpool.terminate() # clear the pool
    del mainpool # delete pool
    return out
#except:
#  print("Multiprocessing not found, running in a single core")
#  def pcall_mp(fun,args,cores=1): return pcall_serial(fun,args)
  





def pcall(fun,args): # define the function
  global cores,is_child
  from multiprocessing import current_process
  if not is_child: # if main process
#  if current_process().name == 'MainProcess': # main process
    is_child = True # child from now on
    if cores==1: out = pcall_serial(fun,args) # one core, simply iterate
    else: 
        try: out = pcall_mp(fun,args,cores=cores) # call in parallel
        except:
            print("Something wrong happened in the parallel execution")
            out = pcall_serial(fun,args) # serial execution
    is_child = False # main from now on
    return out
  # child process
  else: return pcall_serial(fun,args) # one core, simply iterate

