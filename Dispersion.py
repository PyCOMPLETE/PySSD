class Dispersion:
    _detuning = None;
    _distribution = None;
    _Q = 0.0;
    _epsilon = 1E-6;

    def __init__(self, distribution, detuning, Q, epsilon=1E-6):
        self._distribution = distribution
        self._detuning = detuning
        self._Q = Q
        self._epsilon = epsilon

    def setEpsilon(self,epsilon):
        self._epsilon = epsilon

    def getEpsilon(self):
        return self._epsilon

    def getValue(self, jx, jy):

        # DI = (
        #      (jx * self._distribution.getDJx(jx,jy)) /
        #      (complex(self._Q - self._detuning(jx,jy), self._epsilon))
        #      )
        DI = (
            (jx * self._distribution.getDJx(jx, jy)) /
            (self._Q - self._detuning(jx, jy) + 1j*self._epsilon)
             )

        return DI


class RealDispersion(Dispersion):
    def __call__(self,jx,jy):
        return self.getValue(jx,jy).real


class ImaginaryDispersion(Dispersion):
    def __call__(self,jx,jy):
        return self.getValue(jx,jy).imag