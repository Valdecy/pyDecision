###############################################################################

# Required Libraries
import numpy as np

###############################################################################

# Function: Simplified BWM
def simplified_bw_method(mic, lic, alpha = 0.5,verbose = True):
  ib  = np.argmin(mic)
  iw  = np.argmin(lic)
  wb  = [ 1 for item in mic]
  ww  = [ 1 for item in lic]
  div = 0
  for i in range(0, mic.shape[0]):
    a   = mic[i]
    div = div + 1/a
  wb[ib] = 1/div
  for i in range(0, mic.shape[0]):
    if (i != ib):
      wb[i] = wb[ib]/mic[i]
  ww[iw] = 1/np.sum(lic)
  for i in range(0, lic.shape[0]):
    if (i != iw):
      ww[i] = lic[i]*ww[iw]
  w  = [alpha*wb[i] + (1 - alpha)*ww[i] for i in range(0, mic.shape[0])]
  cr = sum([abs(wb[i] - ww[i]) for i in range(0, mic.shape[0])])
  if (verbose == True):
    print('CR:', np.round(cr, 4))
  return cr, w

###############################################################################