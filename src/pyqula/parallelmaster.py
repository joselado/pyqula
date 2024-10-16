
from .parallelmpi import pcall as pcallmpi
from .parallelmpi import check_mpi

# define the master pcall function

if check_mpi(): # this is an MPI run
    pcall = pcallmpi # MPI call
else:
    from .parallelmultiprocess import pcall as pcallmp
    from .parallelmultiprocess import cores
    pcall = pcallmp # multiprocessing call
