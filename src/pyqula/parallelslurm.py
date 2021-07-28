# this is a special library to launch parallel calculations using slurm

import dill as pickle
import os
from . import filesystem as fs
import signal
import subprocess

srcpath = os.path.dirname(os.path.realpath(__file__)) 

#pickle.settings['recurse'] = True

def pcall(fin,xs,batch_size=1,**kwargs):
    """Wrapper to allow for a batch size"""
    if batch_size==1: return pcall_single(fin,xs,**kwargs)
    else: 
        nx = len(xs) # number of xs
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


def pcall_single(fin,xs,time=10,error=None):
    """Run a parallel calculation with slurm"""
    n = len(xs) # number of calculations
    f = lambda x: fin(x)
    main = "import dill as pickle\nimport os\n"
    main += "import sys ; sys.path.append("+srcpath+")\n"
    main += "try: ii = int(os.environ['SLURM_ARRAY_TASK_ID'])\n"
    main += "except: ii = 0\n"
    main += "f = pickle.load(open('function.obj','rb'))\n"
    main += "v = pickle.load(open('array.obj','rb'))\n"
    main += "folder = 'folder_'+str(ii)\n"
    main += "os.system('mkdir '+folder)\n"
    main += "os.chdir(folder)\n"
    main += "pwd = os.getcwd()\n"
#    main += "out = f(v[ii])\n"
    main += "try: out = f(v[ii])\n"
    main += "except: out = None\n"
    main += "os.chdir(pwd)\n"
    main += "print(out)\n"
    main += "pickle.dump(out,open('out.obj','wb'))\n"
    main += "os.system('touch DONE')\n"
    pfolder = ".parallel"
    fs.rmdir(pfolder) # create directory
    fs.mkdir(pfolder) # create directory

    pickle.dump(f,open(pfolder+"/function.obj","wb")) # write function
    pickle.dump(xs,open(pfolder+"/array.obj","wb")) # write object
    open(pfolder+"/run.py","w").write(main) # write script
    hours = str(int(time)) # hours
    mins = int((time-int(time))*60)
    mins = str(max([mins,1])) # at least 1 minute
    runsh = "#!/bin/bash\n#SBATCH -n 1\n#SBATCH -t "+str(int(time))+":"+str(mins)+":00\n"
    runsh += "#SBATCH --mem-per-cpu=5000\n"
    runsh += "#SBATCH --array=0-"+str(n-1)+"\n"
    runsh += "srun python run.py\n"
    open(pfolder+"/run.sh","w").write(runsh) # parallel file
    pwd = os.getcwd() # current directory 
    os.chdir(pfolder) # go to the folder
#    os.system("sbatch run.sh >> run.out") # run calculation
    out,err = subprocess.Popen(["sbatch","run.sh"],stdout=subprocess.PIPE).communicate()
    job = job_number(out) # job number
    jobkill(job) # kill the job if exiting
    os.chdir(pwd) # back to main
    import time
    from os import path
    time.sleep(0.5) # wait half a second
    while True:
        finished = True
        for i in range(n):
            if not path.exists(pfolder+"/folder_"+str(i)+"/DONE"):
                finished = False
        if finished: break
    # get all the data
    ys = []
    for i in range(n):
        folder = pfolder+"/folder_"+str(i)+"/"
        y = pickle.load(open(folder+'out.obj','rb'))
        if y is None: y = error # use this as backup variable
        ys.append(y)
    return ys



def job_number(out):
    """Get the job number"""
    out = str(out)
    out = out.split("job")[1]
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


