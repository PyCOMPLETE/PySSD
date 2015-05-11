import numpy as np

class Detuning:
    def __call__(self,jx,jy):
        raise NotImplemented;

class LinearDetuning(Detuning):
    _startTune = 0.31;
    _slopex = 0.0;
    _slopey = 0.0;

    def __init__(self,startTune,slopex,slopey):
        self._startTune = startTune
        self._slopex = slopex
        self._slopey = slopey

    def __call__(self,jx,jy):
        return self._startTune + self._slopex*jx+self._slopey*jy


class FootprintDetuning(Detuning):
    _footprint = None;
    _plane = None;

    #0 for H and 1 for V
    def __init__(self,footprint,plane=0):
        self._footprint = footprint
        self._plane = plane

    def __call__(self,jx,jy):
        sigx = np.sqrt(2.0*jx)
        sigy = np.sqrt(2.0*jy)
        return self._footprint.getTunesForAmpl(sigx,sigy)[self._plane]
