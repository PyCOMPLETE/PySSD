import numpy as np
from scipy.interpolate import RectBivariateSpline

from matplotlib import pyplot as plt

class Footprint:
    def __init__(self,fileName,maxJ,nJ,nHeaders=8,plane='X',removeResonances=[1,2,3,4,5],tol=1E-4):
        if plane == 'X' or plane == 'H' or plane == 'x' or plane == 'h':
            index = 2
            index2 = 3
        elif plane == 'Y' or plane == 'V' or plane == 'y' or plane == 'v':
            index = 3
            index2 = 2
        else:
            print('Footprint.py :',plane,'is not a valid plane, abort')
            exit()
        self._jxs = np.arange(0.0,maxJ,maxJ/nJ)
        self._jys = np.arange(0.0,maxJ,maxJ/nJ)
        self._gridX,self._gridY = np.meshgrid(self._jxs,self._jys)
        data = np.genfromtxt(fileName,skip_header=nHeaders)
        self._tunes = np.reshape(data[:,index],np.shape(self._gridX))
        self._tunes2 = np.reshape(data[:,index2],np.shape(self._gridX))

        for order in removeResonances:
            for i in range(nJ):
                for j in range(nJ):
                    if self._onRes(self._tunes[i,j], self._tunes2[i,j], order,tol=tol):
                        self._tunes[i,j] = np.nan
        while(np.any(np.isnan(self._tunes))):
            for i in range(nJ):
                for j in range(nJ):
                    if np.isnan(self._tunes[i,j]):
                        nAvg = 0.0
                        self._tunes[i,j]= 0.0
                        if i+1 < nJ and not np.isnan(self._tunes[i+1,j]):
                            self._tunes[i,j] += self._tunes[i+1,j]
                            nAvg += 1
                        if j+1 < nJ and not np.isnan(self._tunes[i,j+1]):
                            self._tunes[i,j] += self._tunes[i,j+1]
                            nAvg += 1
                        if i-1 > 0 and not np.isnan(self._tunes[i-1,j]):
                            self._tunes[i,j] += self._tunes[i-1,j]
                            nAvg += 1
                        if j-1 > 0 and not np.isnan(self._tunes[i,j-1]):
                            self._tunes[i,j] += self._tunes[i,j-1]
                            nAvg += 1
                        if nAvg > 0:
                            self._tunes[i,j]/=nAvg
                        else:
                            self._tunes[i,j] = np.nan

        self._interpolator = RectBivariateSpline(self._jxs,self._jys,self._tunes)
        self._interpolator2 = RectBivariateSpline(self._jxs,self._jys,self._tunes2)

    def _onRes(self,qx,qy,order,tol=1E-4):
        for n in range(1,order+1):
            m = order-n
            value = (n*qx+m*qy)
            if np.abs(value-np.floor(value+0.5))<tol:
                return True
            value = (n*qx-m*qy)
            if np.abs(value-np.floor(value+0.5))<tol:
                return True
        return False

    def interpolateTunesForAction(self,jx,jy):
        return np.transpose(self._interpolator(jx,jy)) # transpose to obtain the same convention as meshgrid

