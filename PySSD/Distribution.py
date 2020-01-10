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

    def __init__(self,emitx=1.0,emity=1.0):
        self.emitx = emitx
        self.emity = emity

    def getValue(self, jx, jy):
        '''Return pdf of normal 2D Gaussian'''
        return np.exp(-(jx/self.emitx + jy/self.emity))/(self.emitx*self.emity)

    def getDJx(self, jx, jy):
        return -self.getValue(jx, jy)/self.emitx

    def getDJy(self, jx, jy):
        return -self.getValue(jx, jy)/self.emity
