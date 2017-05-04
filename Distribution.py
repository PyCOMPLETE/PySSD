import numpy as np

from abc import ABCMeta, abstractmethod


class Distribution(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getValue(self, jx, jy):
        '''Return local pdf'''
        pass

    @abstractmethod
    def getDJx(self, jx, jy):
        '''Return derivation by Jx'''
        pass

    @abstractmethod
    def getDJy(self, jx, jy):
        '''Return derivation by Jy'''
        pass


class Gaussian(Distribution):

    def getValue(self, jx, jy):
        '''Return pdf of normal 2D Gaussian'''
        return np.exp(-(jx + jy))

    def getDJx(self, jx, jy):
        return -self.getValue(jx, jy)

    def getDJy(self, jx, jy):
        return -self.getValue(jx, jy)
