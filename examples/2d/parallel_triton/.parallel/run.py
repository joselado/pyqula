import dill as pickle
import os
os.system('touch START')
import sys ; sys.path.append('/home/joselado/Documents/programs/pyqula/src/pyqula/..')
try: ii = int(os.environ['SLURM_ARRAY_TASK_ID'])
except: ii = 0
f = pickle.load(open('function.obj','rb'))
v = pickle.load(open('array.obj','rb'))
folder = 'folder_'+str(ii)
os.system('mkdir '+folder)
os.chdir(folder)
pwd = os.getcwd()
try: out = f(v[ii])
except: out = None
os.chdir(pwd)
print(out)
pickle.dump(out,open('out.obj','wb'))
os.system('touch DONE')
