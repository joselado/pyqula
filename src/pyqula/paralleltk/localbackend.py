# local backend for parallelslurm.py: run the tasks written by
# clustercommon.build_job as plain local subprocesses, so that code
# written against parallelslurm.pcall also works on a laptop that has
# no sbatch. Each task is launched exactly as Slurm's "srun python
# run.py" would, just with SLURM_ARRAY_TASK_ID set by hand instead of
# by the scheduler, so run.py itself needs no changes.

import os
import sys
import subprocess
import signal
import time


def submit_and_wait(pfolder,n,cores=None):
    """Run n tasks as local subprocesses, at most `cores` at a time,
    and block until all are done"""
    if cores is None: cores = os.cpu_count() or 1
    cores = max(1,int(cores))

    procs = {} # task index -> Popen
    def killall(*args):
        for p in procs.values():
            if p.poll() is None: p.terminate()
        print("Local jobs killed")
        exit()
    signal.signal(signal.SIGINT, killall)
    signal.signal(signal.SIGTERM, killall)

    next_i = 0
    finished = set()
    while len(finished)<n:
        while next_i<n and (len(procs)-len(finished))<cores:
            env = dict(os.environ)
            env["SLURM_ARRAY_TASK_ID"] = str(next_i)
            procs[next_i] = subprocess.Popen([sys.executable,"run.py"],
                    cwd=pfolder,env=env)
            next_i += 1
        time.sleep(0.2)
        for i,p in procs.items():
            if i not in finished and p.poll() is not None:
                finished.add(i)
