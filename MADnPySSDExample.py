
import sys,os
import numpy as np

from matplotlib import pyplot as plt

from PySSD.Detuning import FootprintDetuning
from PySSD.Distribution import Gaussian
from PySSD.Integrator import Integrator
from PySSD.Footprint import Footprint

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


if __name__ == '__main__':
    inputFileName = './MADnPySSDExample/dynaptune'
    outputFileName = 'MADnPySSDExample.sdiag'
    footprint = Footprint(inputFileName,nJ=20,maxJ=5,plane='H') # nJ and maxJ need to match the parameters for the generation of the footprint in madx (at the end of footprint.mad)
    detuning = FootprintDetuning(footprint)
    distrib = Gaussian()
    maxJ = 4.5 # 3 sigma
    integrator = Integrator(distrib,detuning,maxJ=maxJ)
    
    Qs = findQs(detuning,maxJ=maxJ,nStep=100)
    tuneShifts = []
    myFile = open(outputFileName,'w')
    endLine = ''
    for i in range(len(Qs)):
        print(i,'/',len(Qs))
        tuneShift = integrator.integrate(Qs[i])
        tuneShifts.append(tuneShift)
        if i==0:
            myFile = open(outputFileName,'w')
        else:
            myFile = open(outputFileName,'a')
            myFile.write('\n')
        myFile.write(str(tuneShift.real)+','+str(tuneShift.imag))
        myFile.close()

    plt.figure(0)
    plt.plot(np.real(tuneShifts),np.imag(tuneShifts))
    plt.xlabel('Real tune shift')
    plt.ylabel('Imaginargy tune shift')
    plt.grid()
    plt.show()
    
