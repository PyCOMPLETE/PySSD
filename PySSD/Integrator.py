from __future__ import division

import numpy as np
from PySSD.Detuning import LinearDetuning,FootprintDetuning,PolarFootprintDetuning

from matplotlib import pyplot as plt

class Integrator:
    def __init__(self, distribution, detuning,minJ=0, maxJ=18, nStep=2000,epsilon=1E-6,plane='H'):
        dx = (maxJ - minJ)/nStep
        dy = (maxJ - minJ)/nStep
        jx = np.arange(minJ, maxJ, dx)
        jy = np.arange(minJ, maxJ, dy)
        gridX,gridY = np.meshgrid(jx,jy)
        if plane == 'X' or plane == 'H' or plane == 'x' or plane == 'h':
            self._jdPsi = gridX*distribution.getDJx(gridX, gridY)
        elif plane == 'Y' or plane == 'V' or plane == 'y' or plane == 'v':
            self._jdPsi = gridY*distribution.getDJy(gridX, gridY)
        else:
            print('Integrator.py :',plane,'is not a valid plane, abort')
            exit()

        self._jdPsi*=dx*dy

        if isinstance(detuning,PolarFootprintDetuning):
            self._detuning = np.zeros_like(gridX,dtype=complex)
            for i in range(np.shape(gridX)[0]):
                for j in range(np.shape(gridX)[1]):
                    self._detuning[i,j] = detuning(gridX[i,j],gridY[i,j]) - 1j*epsilon
            #print('detuning from polar footprint',self._detuning)
        elif isinstance(detuning,FootprintDetuning):
            self._detuning = detuning(jx,jy) - 1j*epsilon
            #print('detuning from cartesian footprint',self._detuning)
        else:
            self._detuning = detuning(gridX,gridY) - 1j*epsilon

    def integrate(self, Q):
        values = self._jdPsi / (Q - self._detuning)
        dispersionIntegral = np.sum(values[1:-1,1:-1])
        dispersionIntegral += 0.5*(np.sum(values[0,1:-1]) + np.sum(values[-1,1:-1]) + np.sum(values[1:-1,0]) + np.sum(values[1:-1,-1]))
        dispersionIntegral += 0.25*(values[0,0]+values[0,-1]+values[-1,0]+values[-1,-1])
        return -1.0/dispersionIntegral
