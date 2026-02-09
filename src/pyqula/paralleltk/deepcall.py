def pcall(fun,args,cores=None):
    """Perform a parallel call at the depest level, assuming
    that there is no other potential parallelization inside the
    function"""
    from ..parallel import pcall_serial
    from multiprocessing import current_process
    if cores is None: # if none given, take it from the environment variable
        from ..parallel import cores
    if cores==1:
        return pcall_serial(fun,args)
    else:
        from pathos.multiprocessing import ProcessPool
        with ProcessPool(nodes=cores) as pool:
            out = pool.map(fun,args)
        return out

