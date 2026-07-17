# this is a special library to launch parallel calculations, one
# process per input, using Slurm array jobs on a cluster (where sbatch
# is available) or plain local subprocesses otherwise (e.g. a laptop)

import shutil

from .paralleltk import clustercommon
from .paralleltk import slurmbackend
from .paralleltk import localbackend


def pcall(fin,xs,batch_size=1,**kwargs):
    """Wrapper to allow for a batch size"""
    if batch_size==1: return pcall_single(fin,xs,**kwargs)
    else:
        xsn = [] # empty list
        o = []
        for i in range(len(xs)):
            o.append(xs[i]) # store
            if i%batch_size==0: # reached the limit
                xsn.append(o) # store
                o = [] # reset
        def fnew(y): return [fin(x) for x in y] # call this batch
        outs = pcall_single(fnew,xsn,**kwargs) # call the inputs
        out = []
        for o in outs: out += o # add
        return out


def pcall_killproof_dict(fin,xs,info=True,**kwargs):
    """Call method that is relaunched for killed jobs"""
    outl = pcall_single(fin,xs,**kwargs) # the return is a list
    out = dict()
    for i in range(len(xs)):
        out[xs[i]] = outl[i] # store
    xsnew = [] # empty list
    for (x,o) in zip(xs,out): # loop over keys
        if out[o] is None: # this one has been killed/failed
            if info:
                print("Relaunching",o)
            xsnew.append(o) # store
    if len(xsnew)==0:
        return out # all good
    else:
        out2 = pcall_killproof_dict(fin,xsnew,info=info,**kwargs) # new outputs
        for o in out2:
            out[o] = out2[o] # overwrite
        return out

def pcall_killproof(fin,xs,return_mode="list",**kwargs):
    out = pcall_killproof_dict(fin,xs,**kwargs)
    if return_mode=="list":
        return [out[x] for x in xs]
    elif return_mode=="dict":
        return out


def has_slurm():
    """Whether this machine can submit Slurm array jobs"""
    return shutil.which("sbatch") is not None


def pcall_single(fin,xs,time=10,memory=5000,error=None,
    constraint=None,cores=None,backend="auto",
    return_mode="list"):
    """Run one process per input, through Slurm if available
    (backend="slurm") or local subprocesses otherwise (backend="local");
    backend="auto" (the default) picks the former if sbatch exists"""
    n = len(xs) # number of calculations
    pfolder = ".parallel"
    clustercommon.build_job(fin,xs,pfolder) # write function/inputs/run.py

    if backend=="auto":
        backend = "slurm" if has_slurm() else "local"

    if backend=="slurm":
        slurmbackend.submit_and_wait(pfolder,n,time_=time,memory=memory,
                constraint=constraint)
    elif backend=="local":
        localbackend.submit_and_wait(pfolder,n,cores=cores)
    else:
        raise ValueError("Unknown backend '"+str(backend)+"'")

    ys = clustercommon.collect_results(pfolder,n,error=error)
    if return_mode=="list": return ys
    elif return_mode=="dict":
      outys = dict() # dictionary
      for i in range(n): outys[xs[i]] = ys[i]
      return outys # return the dictionary
