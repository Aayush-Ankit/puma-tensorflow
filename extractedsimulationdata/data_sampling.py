import numpy as np
import sys
data1=np.loadtxt(sys.argv[1])
#Filtering
data1_samp=data1[::int(sys.argv[2])]
np.savetxt(sys.argv[3],data1_samp)
