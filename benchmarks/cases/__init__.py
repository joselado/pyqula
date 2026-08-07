from . import diag_dense_sparse, dos_methods, green_numba

CASES = {
    green_numba.CASE_NAME: green_numba,
    dos_methods.CASE_NAME: dos_methods,
    diag_dense_sparse.CASE_NAME: diag_dense_sparse,
}
