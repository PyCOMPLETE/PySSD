from __future__ import division

import warnings
import numpy as np
from scipy.integrate import dblquad, simps

from abc import ABCMeta, abstractmethod

from PySSD.Dispersion import Dispersion, RealDispersion, ImaginaryDispersion


class Integrator(object):

    __metaclass__ = ABCMeta

    def __init__(self, distribution, detuning, minJ=0.0, maxJ=18.0):
        self.distribution = distribution
        self.detuning = detuning
        self.minJx = minJ
        self.maxJx = maxJ
        self.minJy = minJ
        self.maxJy = maxJ

    @abstractmethod
    def integrate(self): pass


class Boundary:
    _boundary = 0.0

    def __init__(self, boundary):
        self._boundary = boundary

    def __call__(self, x):
        return self._boundary


class DblquadIntegrator(Integrator):

    def __init__(self, distribution=None, detuning=None, minJ=0.0, maxJ=18.0):

        self.distribution = distribution
        self.detuning = detuning
        self.minJx = minJ
        self.maxJx = maxJ
        self.minJy = Boundary(minJ)
        self.maxJy = Boundary(maxJ)

    def integrate(self, Q):

        rDispersion = RealDispersion(self.distribution, self.detuning, Q)
        iDispersion = ImaginaryDispersion(self.distribution, self.detuning, Q)

        realPart, realErr = dblquad(rDispersion,
                                    self.minJx, self.maxJx,
                                    self.minJy, self.maxJy)
        imagPart, imagErr = dblquad(iDispersion,
                                    self.minJx, self.maxJx,
                                    self.minJy, self.maxJy)

        return -1.0/complex(realPart, imagPart)


class SimpsonIntegrator(Integrator):

    def __init__(self, *args, **kwargs):

        super(SimpsonIntegrator, self).__init__(*args, **kwargs)

        if 'n_steps' not in kwargs:
            kwargs['n_steps'] = 1000
        n_steps = kwargs['n_steps']
        self.jx = np.linspace(self.minJx, self.maxJx, n_steps)
        self.jy = np.linspace(self.minJy, self.maxJy, n_steps)
        self.JX, self.JY = np.meshgrid(self.jx, self.jy)

    def integrate(self, Q, epsilon=1e-6):

        dd = Dispersion(
            self.distribution, self.detuning, Q, epsilon=epsilon
        ).getValue(self.JX, self.JY)
        # dd = np.array([[
        #     Dispersion(
        #         self.distribution, self.detuning, Q, epsilon=epsilon
        #     ).getValue(x, y)
        #     for y in self.jy] for x in self.jx])

        return -1./simps(simps(dd, self.jx), self.jy)


class TrapzIntegrator(Integrator):

    def __init__(self, *args, **kwargs):

        super(TrapzIntegrator, self).__init__(*args, **kwargs)

        if 'n_steps' not in kwargs:
            kwargs['n_steps'] = 1000
        n_steps = 1000
        self.jx = np.linspace(self.minJx, self.maxJx, n_steps)
        self.jy = np.linspace(self.minJy, self.maxJy, n_steps)
        self.JX, self.JY = np.meshgrid(self.jx, self.jy)

    def integrate(self, Q, epsilon=1e-6):

        dd = Dispersion(
            self.distribution, self.detuning, Q, epsilon=epsilon
        ).getValue(self.JX, self.JY)

        return -1./np.trapz(np.trapz(dd, self.jx), self.jy)


class FixedTrapezoidalIntegrator(Integrator):

    def __init__(self, distribution=None, detuning=None,
                 minJ=0, maxJ=18, nStep=2000):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('Class FixedTrapezoidalIntegrator' +
                      'is deprecated and will be replaced in the ' +
                      'near future.',
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)

        self._distribution = distribution
        self._detuning = detuning
        self._minJx = minJ
        self._maxJx = maxJ
        self._minJy = minJ
        self._maxJy = maxJ
        self._nStep = nStep

    def integrate(self, Q, epsilon=1e-6):

        dispersion = Dispersion(self._distribution, self._detuning, Q, epsilon=epsilon)

        dx = (self._maxJx - self._minJx)/self._nStep
        dy = (self._maxJy - self._minJy)/self._nStep
        arrayX = np.arange(self._minJx, self._maxJx, dx)
        arrayY = np.arange(self._minJy, self._maxJy, dy)

        dispersionIntegral = 0.25 * (
            dispersion.getValue(arrayX[0], arrayY[0])  +
            dispersion.getValue(arrayX[0], arrayY[-1]) +
            dispersion.getValue(arrayX[-1], arrayY[0]) +
            dispersion.getValue(arrayX[-1],arrayY[-1]))

        for i in range(1,len(arrayX)-1):
            dispersionIntegral += 0.5*dispersion.getValue(arrayX[i],arrayY[0]);
            dispersionIntegral += 0.5*dispersion.getValue(arrayX[i],arrayY[-1]);
        for i in range(1,len(arrayY)-1):
            dispersionIntegral += 0.5*dispersion.getValue(arrayX[0],arrayY[i]);
            dispersionIntegral += 0.5*dispersion.getValue(arrayX[-1],arrayY[i]);
        for i in range(1,len(arrayX)-1):
            for j in range(1,len(arrayY)-1):
                dispersionIntegral += dispersion.getValue(arrayX[i],arrayY[j]);

        dispersionIntegral *= dx*dy;

        return -1.0/dispersionIntegral;


class AdaptiveRectangularIntegrator(Integrator):

    _detuning = None;
    _distribution = None;
    _minJx = 0.0;
    _maxJx = 0.0;
    _minJy = None;
    _maxJy = None;

    _basis = 2;
    _initialSize = 200;
    _maxIter = 2;
    _maxSize = None;

    _fValues = None;
    _diffX = None;
    _diffY = None;
    _arrayX = None;
    _arrayY = None;


    def __init__(self, distribution, detuning, minJ=0.0, maxJ=18.0):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn('Class "{:s}" '.format(self.__name__) +
                      'is deprecated and will be replaced in the ' +
                      'near future.',
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)

        self._distribution = distribution;
        self._detuning = detuning;
        self._minJx = minJ;
        self._maxJx = maxJ;
        self._minJy = minJ;
        self._maxJy = maxJ;

        self._maxSize = self._initialSize*self._basis**(self._maxIter);

        minDx = (self._maxJx-self._minJx)/self._maxSize;
        minDy = (self._maxJy-self._minJy)/self._maxSize;
        self._arrayX = np.arange(self._minJx,self._maxJx,minDx);
        self._arrayY = np.arange(self._minJy,self._maxJy,minDy);

    def integrate(self,Q):
        fValues = spm.dok_matrix((self._maxSize,self._maxSize),dtype=complex);
        diffX = spm.dok_matrix((self._maxSize,self._maxSize));
        diffY = spm.dok_matrix((self._maxSize,self._maxSize));
        dispersion = Dispersion(self._distribution,self._detuning,Q);
        stepSize = self._maxSize/self._initialSize;
        indices = np.arange(0,self._maxSize,stepSize);
        dx = self._arrayX[stepSize]-self._arrayX[0];
        dy = self._arrayY[stepSize]-self._arrayY[0];
        for i in indices:
            for j in indices:
                fValues[i,j] = dispersion.getValue(self._arrayX[i],self._arrayY[j]);
                diffX[i,j] = dx;
                diffY[i,j] = dy;

        for nIter in np.arange(self._maxIter):
            oldStepSize = stepSize;
            stepSize = self._maxSize/(self._basis**(nIter+1)*self._initialSize);
            print(nIter,fValues.nnz);
            dx = self._arrayX[stepSize]-self._arrayX[0];
            dy = self._arrayY[stepSize]-self._arrayY[0];
            median = np.median(np.abs(fValues.values()));
            for key in fValues.keys():
                if np.abs(fValues[key]) > median:
                    fValues[key] = 0.0;
                    diffX[i,j] = 0.0;
                    diffY[i,j] = 0.0;
                    for i in np.arange(key[0]-oldStepSize/2,key[0]+oldStepSize/2,stepSize):
                        for j in np.arange(key[1]-oldStepSize/2,key[1]+oldStepSize/2,stepSize):
                            fValues[i,j] = dispersion.getValue(self._arrayX[i],self._arrayY[j]);
                            diffX[i,j] = dx;
                            diffY[i,j] = dy;

        dispersionIntegral = complex(0.0,0.0);
        for key in fValues.keys():
            dispersionIntegral += fValues[key]*diffX[key]*diffY[key];

        print('integral',dispersionIntegral)

        return -1.0/dispersionIntegral;
