# Slurm backend for parallelslurm.py: submit the tasks written by
# clustercommon.build_job as a job array via sbatch, wait for them to
# finish (or be killed), and scancel the job if this process is killed

import os
import subprocess
import signal
import time
from os import path


def submit_and_wait(pfolder,n,time_=10,memory=5000,constraint=None):
    """Submit an array job of n tasks and block until all are done"""
    hours = str(int(time_))
    mins = int((time_-int(time_))*60)
    mins = str(max([mins,1])) # at least 1 minute
    runsh = "#!/bin/bash\n#SBATCH -n 1\n#SBATCH -t "+str(int(time_))+":"+str(mins)+":00\n"
    runsh += "#SBATCH --mem-per-cpu="+str(memory)+"\n"
    runsh += "#SBATCH --exclude=milan[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]\n"
    from ..parallel import numba_cores
    if numba_cores is None: cores = 1
    else: cores = numba_cores
    runsh += "#SBATCH --cpus-per-task="+str(cores)+"\n"
    runsh += "#SBATCH --array=0-"+str(n-1)+"\n"
    if constraint is not None:
        runsh += "#SBATCH --constraint="+str(constraint)+"\n"
    runsh += "srun python run.py\n"
    open(pfolder+"/run.sh","w").write(runsh) # parallel file
    pwd = os.getcwd() # current directory
    os.chdir(pfolder) # go to the folder
    out,err = subprocess.Popen(["sbatch","run.sh"],stdout=subprocess.PIPE).communicate()
    job = job_number(out) # job number
    jobkill(job) # kill the job if exiting
    os.chdir(pwd) # back to main
    time.sleep(0.5) # wait half a second
    while True:
        finished = True
        time.sleep(0.5) # wait half a second
        for i in range(n):
            pfolderi = pfolder+"/folder_"+str(i)
            if started_and_killed(pfolderi,str(job)+"_"+str(i)):
                pass # ignore as if it finished
            else:
                if not path.exists(pfolderi+"/DONE"):
                    finished = False
        if finished: break
        # check if it has been killed


def job_number(out):
    """Get the job number"""
    out = str(out)
    out = out.split("Submitted batch job")[1]
    out = out.split("\\n")[0]
    return int(out) # return the job


def jobkill(n):
    """Kill the job when the program is killed"""
    def killf(*args):
      subprocess.Popen(["scancel",str(n)],stdout=subprocess.PIPE).communicate()
      print("Job killed")
      exit()
    signal.signal(signal.SIGINT, killf)
    signal.signal(signal.SIGTERM, killf)


def started_and_killed(inipath,number):
    """Check if a certain job that was started has been killed"""
    if path.exists(inipath+"/START"): # the job started
        out,err = subprocess.Popen(["squeue","-r"],
                      stdout=subprocess.PIPE).communicate()
        out = out.decode("utf-8").split("\n")
        for o in out:
            try:
                if number in o:
                    return False
            except: pass
        return True # the job was killed
    else: return False # the job has not started
