# shared on-disk protocol used by both the Slurm and local backends of
# parallelslurm.py: pickle the function/inputs, write the per-task
# run.py script, and collect the results it writes back

import os
import dill as pickle
from .. import filesystem as fs

srcpath = os.path.dirname(os.path.realpath(__file__))+"/../.."


def build_job(fin,xs,pfolder):
    """Write function.obj/array.obj and the run.py that each array task
    (task index ii, read from SLURM_ARRAY_TASK_ID, defaulting to 0)
    executes to produce folder_ii/out.obj"""
    main = "import dill as pickle\nimport os\n"
    main += "os.system('touch START')\n"
    main += "import sys ; sys.path.append('"+srcpath+"')\n"
    main += "try: ii = int(os.environ['SLURM_ARRAY_TASK_ID'])\n"
    main += "except: ii = 0\n"
    main += "f = pickle.load(open('function.obj','rb'))\n"
    main += "v = pickle.load(open('array.obj','rb'))\n"
    main += "folder = 'folder_'+str(ii)\n"
    main += "os.system('mkdir '+folder)\n"
    main += "os.chdir(folder)\n"
    main += "pwd = os.getcwd()\n"
    main += "try: out = f(v[ii])\n"
    main += "except: out = None\n"
    main += "os.chdir(pwd)\n"
    main += "print(out)\n"
    main += "pickle.dump(out,open('out.obj','wb'))\n"
    main += "os.system('touch DONE')\n"
    fs.rmdir(pfolder) # create directory
    fs.mkdir(pfolder) # create directory
    pickle.dump(fin,open(pfolder+"/function.obj","wb")) # write function
    pickle.dump(xs,open(pfolder+"/array.obj","wb")) # write object
    open(pfolder+"/run.py","w").write(main) # write script


def collect_results(pfolder,n,error=None):
    """Read back folder_i/out.obj for each of the n tasks"""
    ys = []
    for i in range(n):
        folder = pfolder+"/folder_"+str(i)+"/"
        try: y = pickle.load(open(folder+"out.obj","rb"))
        except: y = None # in case the file does not exist
        if y is None: y = error # use this as backup variable
        ys.append(y)
    return ys
