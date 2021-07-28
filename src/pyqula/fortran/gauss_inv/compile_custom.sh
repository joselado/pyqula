f2py2.7 -llapack -c -m gauss_inv gauss_inv.f90
cp gauss_inv.so ../../ # copy to the main directory
