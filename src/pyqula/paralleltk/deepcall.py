def pcall(fun,args):
    """Perform a parallel call at the depest level, assuming
    that there is no other potential parallelization inside the
    function"""
    from ..parallel import cores,pcall_serial
    if cores==1:
        return pcall_serial(fun,args)
    else:
        from pathos.multiprocessing import ProcessPool
        with ProcessPool(nodes=cores) as pool:
            out = pool.map(fun,args)
        return out

