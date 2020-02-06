import os,sys

import math
import collections
import numpy as np

sys.path.append("/pymorph")
from schemes.weno import get_left_flux,get_right_flux,get_Max_Phase_Speed
from schemes.weno import get_stencil
import sediment_transport.sed_trans as sedtrans
import morph_geom_lib as mgl

import models.simple_depth_morph_models as simple_models
import models.shallow_depth_morph_models as shallow_models

from schemes.avalanche_scheme import *


class ParameterizedMorphologicalModel(shallow_models.NullShallowHydroMorphologicalModel):

    def set_bedload_model(self, bedloadModel):
        self._bedloadModel = bedloadModel


    def _calculate_bedload(self, h, u, xc, z, a=None, b=None):
        return self._bedloadModel.calculate_bedload(h, u, xc, z, self._time)

    def run(self, simulationTime, dt, extractionTime, fileName):

        print(' Starting simulation....')
        # --------------------------------
        #  Setup the model run parameters
        # --------------------------------
        nt = int(simulationTime / dt)  # Number of time steps
        print('Number of time steps: {0} mins'.format(nt/60.))
        slope = np.gradient(self._zc, self._dx)
        
        self._a = 0
        self._b = 0

        # --------------------------------
        # Set up the domain, BCs and ICs
        # --------------------------------
        print('Grid dx = {0}'.format(self._dx))
        print('Grid nx = {0}'.format(self._nx))
        
        # --------------------------------
        # Initialize the hydro model transport
        # --------------------------------
        print('Initializing hydrodynamic model...')
       
        h  = np.zeros(self._nx)
        u = np.zeros(self._nx)
        q = np.zeros(self._nx)
        
        print('Completed the intialization of the model')

        print('D50:    {0}'.format(self._D50))
        print('Rho Particle:    {0}'.format(self._rho_particule))
        print('Angle Repose Degrees:    {0}'.format(self._repose_angle))
        # --------------------------------
        # Initialize the sed transport
        # --------------------------------
        
        qbedload = self._bedloadModel.calculate_bedload(self._h, self._u, self._xc, self._zc, self._time)
      

        # --------------------------------
        #  Run the model
        # --------------------------------
        cntr = 0
        
        xmax = self._xc.max()
        resolution_cells = len(self._xc)
        for n in range(1, nt):
            
            time = n*dt
            
            self._time = time
            
            znp1 = np.zeros(self._nx)

            # --------------------------------
            # Update the bed
            # --------------------------------
            znp1 = self._exner_model.update_bed(self._zc, qbedload, dt, self)
            
            # --------------------------------
            # Avalanche the bed
            # --------------------------------
            if self._useAvalanche == True:
                znp1 = self._avalanche_model(self._xc, znp1)
            
            # --------------------------------
            # Appy smoothing filter
            # --------------------------------
            if self._useSmoother == True:
                znp1 = self._apply_smoothing_filter(self._xc, znp1)
                
            # Apply dune length correction
            '''a = 0.001666667
            b = 0.01
            
            m = b/(b + math.exp(-1.*b*time))
            xnew = np.linspace(0,xmax*0.5**m, len(self._zc))
            #f = interp1d(xadj, self._zc, fill_value="extrapolate")
            
            #xnew = np.linspace(0., xmax, num=resolution_cells)
            #self._zc = f(xnew)
            self._xc = xnew'''
            
            # --------------------------------
            # update the bedload 
            # --------------------------------
            #qbedload = self._calculate_bedload(self._h, self._u, self._xc, znp1, self._a, self._b)

            qbedload = self._bedloadModel.calculate_bedload(self._h, self._u, self._xc, znp1, self._time)

            self._zc = znp1
            
            
            if (time / extractionTime) == math.floor(time / extractionTime):        
                timestep = n*dt
                self._extract_results(self._xc, self._zc, u, q, h, qbedload, timestep, dt, fileName)              
        return self._xc, self._zc, u, q, h, qbedload
    
'''
This is the abstract class for the bedload model
'''
class NullBedloadModel(object):
    def calculate_bedload(self, h, u, x, z, t):
        pass

class EquilibriumBedloadModel(NullBedloadModel):

    def __init__(self, qsb_max, delta, z_offset):
        self.__qsb_max = qsb_max
        self.__delta = delta
        self.__z_offset = z_offset

    def calculate_bedload(self, h, u, x, z, t):
        znorm = np.array([zs - self.__z_offset for zs in z])
        znorm = znorm.clip(min=0.)
        
        qbedload = [((zs/self.__delta) * self.__qsb_max) for zs in znorm]
        return  qbedload

'''
This model should decay the dune height
'''
class NonEquilibriumBedloadModel(NullBedloadModel):

    def __init__(self, qsb_max, delta, z_offset, c, d):
        self.__qsb_max = qsb_max
        self.__delta = delta
        self.__z_offset = z_offset
        self.__c = c
        self.__d = d
        print('Initalized')

    def calculate_bedload(self, h, u, x, z, t):

        znorm = np.array([zs - self.__z_offset for zs in z])
        znorm = znorm.clip(min=0.)
        zmean = znorm.max() - znorm.min()

        decay = self.__c * math.exp(self.__d*t/60.) + 1
        #print('Decay: ', decay)
        qbedload = [self.__qsb_max*( zs/ self.__delta )**decay for zs in znorm]

        return qbedload

class NonEquilibriumBedloadModel2(NullBedloadModel):

    def __init__(self, qsb_max, delta, z_offset, c, d):
        self.__qsb_max = qsb_max
        self.__delta = delta
        self.__z_offset = z_offset
        self.__c = c
        self.__d = d

    def calculate_bedload(self, h, u, x, z, t):
        znorm = np.array([zs - self.__z_offset for zs in z])
        znorm = znorm.clip(min=0.)
        
        #decay = (1 - self.__c*math.exp(self.__d*t/60.))
        #qbedload = [self.__qsb_max*(z - (z**self.__d)/self. for zs in z]
        qbedload = [self.__qsb_max*(((zs)/self.__delta)) - 0.15 * self.__qsb_max*(zs/self.__delta)**7    for zs in z]

        return qbedload

'''
This is a working version of a bed load model
'''
class DummyBedloadModel(NullBedloadModel):
    def __init__(self,a, b):
        self._a = a
        self._b = b

    def calculate_bedload(self, h, u, x, z, t):

        qbedload = np.zeros(len(x))

        a = 0.0000127  # qbed max
        b = 2
        c = 1.25
        d = -0.025

        # So the dune height is 7.9 cm - make the q - 0.00001 at the max height
        # This will essentially calibrate the model

        # For 32 cm case
        # qbedload = [((zs - 0.0134)/0.079 * 0.0000127) for zs in z]

        #t = self._time / 60.
        m = (c * math.exp(d * (t)) + 1)

        znorm = np.array([(zs - 0.0034) for zs in z])
        znorm = znorm.clip(min=0.)
        zmean = znorm.max() - znorm.min()

        if znorm.min() < 0.:
            print('Error: {0}'.format(znorm.min()))

        # qbedload =  np.array([ a * pow(zs,m) / zmean for zs in znorm])
        # Are the numbers you are attempting to exponentiate ever negative?
        # If so, are you aware that raising a negative number to a non-integral power is undefined (or at least ventures into the realm of complex numbers)? –

        # Apply dune length correction
        d = 0.001666667
        e = 0.01
        n = e / (e + math.exp(-1. * d * self._time))

        start = (2. - n) / 1.

        multiplier = np.linspace(start, 1, len(x))

        for i in range(len(znorm)):
            qbedload[i] = multiplier[i] * ((a * znorm[i] ** m) / 0.036)  # + a *(x[i]/12.)**((1.-n)/1.)

        # For 20 cm case
        # qbedload = [(zs/0.079 * 0.005)**2. for zs in z]

        return qbedload