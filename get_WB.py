# read tb.dat file using wannierberri
# dump all necessary data to a npz file

import glob
# set num of threads to 2
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# load wannierberri package
import wannierberri as wberri
print (f"Using WannierBerri version {wberri.__version__}")
import numpy as np
import scipy
import matplotlib.pyplot as plt
from termcolor import cprint

# using  all cpus
try:
    parallel.shutdown()
except NameError:
    pass
parallel = wberri.Parallel()


# Get the parent directory
parent_dir = os.path.dirname(os.getcwd())

# Use glob to find all _tb.dat files
tb_files = glob.glob(os.path.join(parent_dir, "*_tb.dat"))



# print num of tb_files, should be 1 in usual case
print (f"Found {len(tb_files)} tb.dat files")
# print them
print (f"They are:  {tb_files}")
# use the first one
tb_file = tb_files[0]
print(f"Use {tb_file}")


# initialize the system
system = wberri.System_tb(f"{tb_file}",berry=True)


# all necessary data
HR = system._XX_R['Ham']
cRvec = system.rvec.cRvec
real_lattice = system.real_lattice
recip_lattice = system.recip_lattice
Wan_center = system.wannier_centers_cart
AA_R =system._XX_R['AA']
iRvec = system.rvec.iRvec

# pack all data into a npz file
np.savez("TB.npz",
    HR=HR,
    cRvec=cRvec,
    real_lattice=real_lattice,
    recip_lattice=recip_lattice,
    Wan_center=Wan_center,
    AA_R=AA_R,
    iRvec=iRvec,
)


print("TB.npz saved")
