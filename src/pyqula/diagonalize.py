import scipy.linalg as lg


## routines for diagonalization ##

error = 1e-7

def eigh(m):
    if np.max(np.abs(m.imag))<error: # assume real
        print("Real")
        return lg.eigh(m.real) # diagonalize real matrix
    else: return lg.eigh(m) # diagonalize complex matrix


