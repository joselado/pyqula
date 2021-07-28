f2py2.7  -c -m chi_fortran calculate_chi.f90 
f2py2.7  -c -m collinear_xychi collinear_xychi.f90 
cp collinear_xychi.so ../../
