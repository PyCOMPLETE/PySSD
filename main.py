from __future__ import division

import sys, time
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

from PySSD.Distribution import BiGaussian
from PySSD.Detuning import LinearDetuning
from PySSD.Dispersion import Dispersion
from PySSD.Integrator import FixedTrapezoidalIntegrator, SimpsonIntegrator, TrapzIntegrator

# from PySSD.Detunign import FootprintDetuning
# from PySSD.Footprint import Footprint, parseDynapTune


def findQs(detuning, stepSize=5E-5, maxJ=18.0, dJ=0.1, margin=1):
    '''
    '''
    myMin = 1.0
    myMax = 0.0
    for jx in np.arange(0, maxJ, dJ):
        for jy in np.arange(0, maxJ, dJ):
            value = detuning(jx, jy)
            if value < myMin :
                myMin = value
            if value > myMax :
                myMax = value
    return np.arange(myMin-margin*stepSize, myMax+margin*stepSize, stepSize)


distribution = BiGaussian()

# LINEAR DETUNING
# ===============
energy = 4.0
ROF    = -500
emit   = 2E-6
a =  0.82 * ROF*emit/energy
b = -0.58 * ROF*emit/energy
detuning = LinearDetuning(0.31, a, b)

# FOOTPRINT DETUNING
# ==================
# dynaptune = True; ### TRUE : Use MADX output dynaptune as input. FALSE: Use output from Online Footprint Viewer
# inputFileName = 'dynaptyune_HO.txt'

# if dynaptune:
#     strRep = parseDynapTune(inputFileName, 101, 51); # set number of amplitude and angles in the file. (MADX + 1)
# else:
#     with open(inputFileName, "r") as fid:
#         strRep = fid.readline()

# footprint = Footprint(strRep,dSigma=0.1) # dSigma : amplitude difference between two indices in the amplitude loop (in unit of sigma)
# footprint.repair() # remove faulty points of the footprint
# detuning = FootprintDetuning(footprint)

# fp = footprint.getPlottable()

# plt.plot(fp[0], fp[1], '-')
# plt.show()

maxJ = 18
Qs = findQs(detuning, stepSize=2e-5, maxJ=maxJ)

integrator = FixedTrapezoidalIntegrator(distribution, detuning, maxJ=maxJ)
integrator = SimpsonIntegrator(distribution, detuning, maxJ=maxJ)
integrator = TrapzIntegrator(distribution, detuning, maxJ=maxJ)

tuneShifts = []
outputFileName = './Test.sdiag';

t0 = time.clock()
for i in range(len(Qs)):

    tuneShift = integrator.integrate(Qs[i])
    tuneShifts.append(tuneShift)

    sys.stdout.write("\r{:g}/{:g} - Complex tune shift: {:g}".
                     format(i, len(Qs), tuneShift))

tuneShifts = np.array(tuneShifts)
np.savetxt(outputFileName, [tuneShifts.real, tuneShifts.imag])

t1 = time.clock()-t0
print "Elapsed times: {:g}s.".format(t1)

fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111)

ax.plot([tuneShifts[k].real for k in range(len(tuneShifts))], [tuneShifts[k].imag for k in range(len(tuneShifts))])
ax.set_xlabel("$\operatorname{Re} \Delta Q$", fontsize='large')
ax.set_ylabel("$\operatorname{Im} \Delta Q$", fontsize='large')

plt.show()
