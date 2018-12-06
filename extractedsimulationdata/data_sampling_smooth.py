import numpy as np
import sys
from scipy.signal import savgol_filter

data1=np.loadtxt(sys.argv[1])
#Filtering
#data1_samp=data1[::int(sys.argv[2])]
data1_samp=savgol_filter(data1[::int(sys.argv[2])],51,3)
np.savetxt(sys.argv[3],data1_samp)
