from . import diag_dense_sparse, dos_methods, green_numba, rpa_spin_response

CASES = {
    green_numba.CASE_NAME: green_numba,
    dos_methods.CASE_NAME: dos_methods,
    diag_dense_sparse.CASE_NAME: diag_dense_sparse,
    rpa_spin_response.CASE_NAME: rpa_spin_response,
}
