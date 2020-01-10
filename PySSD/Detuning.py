from __future__ import division

import numpy as np


class Detuning(object):

    def __call__(self, jx, jy):
        raise NotImplemented


class LinearDetuning(Detuning):
    def __init__(self, startTune=0.31, slopex=0, slopey=0):
        self._startTune = startTune
        self._slopex = slopex
        self._slopey = slopey

    def __call__(self, jx, jy):
        return self._startTune + self._slopex*jx + self._slopey*jy


class FootprintDetuning(Detuning):
    def __init__(self, footprint):
        self._footprint = footprint

    def __call__(self, jx, jy):
        return self._footprint.interpolateTunesForAction(jx,jy)

class PolarFootprintDetuning(Detuning):
    # 0 for H and 1 for V
    def __init__(self, footprint=None, plane=0):
        self._footprint = footprint
        self._plane = plane

    def __call__(self, jx, jy):
        '''Actions here are given in units of sigma:
        sigx = x/sigma_x = sqrt(2*Jx*beta_x)/sqrt(eps_x*beta_x)
                         = sqrt(2*Jx/eps_x) = sqrt(2*jx).'''

        if self._plane == 0:
            sigx = np.sqrt(2.0*jx)
            sigy = np.sqrt(2.0*jy)
        else:
            sigy = np.sqrt(2.0*jx)
            sigx = np.sqrt(2.0*jy)

        return self._footprint._getTunesForAmpl(sigx, sigy)[self._plane]
