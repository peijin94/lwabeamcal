"""Explicitly synthetic API demonstration, not an observational validation."""
import numpy as np
from lwabeamcal import calibrate

f=np.array([40.,55.,70.])
raw=np.array([[.6,.65],[.9,.95],[1.2,1.25]])
result=calibrate(raw,f,input_unit='sfu',fast_jy=np.full((3,2),2000.),
                 gain=np.full((3,2),.6),sun_alt_deg=[55.,56.])
print('Corrected Jy (synthetic demonstration):')
print(result.corrected_jy)
print('Valid cells:',int(result.valid.sum()))
