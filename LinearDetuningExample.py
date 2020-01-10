from __future__ import division

import sys, time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

from PySSD.Distribution import Gaussian
from PySSD.Detuning import LinearDetuning
from PySSD.Integrator import Integrator

def findQs(detuning,nStep=20,maxJ=18.0,dJ=0.1,margin=0.1):
    myMin = 1.0
    myMax = 0.0
    for jx in np.arange(0.0,maxJ,dJ):
        for jy in np.arange(0.0,maxJ,dJ):
            value = detuning(jx,jy)
            if value < myMin :
                myMin = value
            if value > myMax :
                myMax = value
    marginQ = margin*(myMax-myMin)
    myMax += marginQ
    myMin -= marginQ
    dQ = (myMax-myMin)/nStep
    return np.arange(myMin,myMax,dQ)


distribution = Gaussian()

# LINEAR DETUNING
# ===============
octDet_direct = 7.77291e-05
octDet_cross = -5.63365e-05
detuning = LinearDetuning(0.31,octDet_direct,octDet_cross)

maxJ = 18 # 6 sigma
Qs = findQs(detuning, nStep = 100, maxJ=maxJ)

integrator = Integrator(distribution, detuning, maxJ=maxJ)

tuneShifts = []
outputFileName = './Test.sdiag';

t0 = time.clock()
for i in range(len(Qs)):

    tuneShift = integrator.integrate(Qs[i])
    tuneShifts.append(tuneShift)

    sys.stdout.write("\r{:g}/{:g} - Complex tune shift: {:g}".
                     format(i+1, len(Qs), tuneShift))

tuneShifts = np.array(tuneShifts)
np.savetxt(outputFileName, [tuneShifts.real, tuneShifts.imag])

t1 = time.clock()-t0
print("Elapsed times: {:g}s.".format(t1))

fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111)

ax.plot([tuneShifts[k].real for k in range(len(tuneShifts))], [tuneShifts[k].imag for k in range(len(tuneShifts))])
ax.set_xlabel("$\operatorname{Re} \Delta Q$", fontsize='large')
ax.set_ylabel("$\operatorname{Im} \Delta Q$", fontsize='large')

plt.show()
