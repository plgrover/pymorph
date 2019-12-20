#! /usr/bin/env python
# -*- coding: utf-8 -*

""" 
    This module contains the shallow-water hydrodynamic model and morphodyamic models.
"""

import math
import schemes.weno as weno
import sediment_transport.sed_trans as sedtrans
from schemes.avalanche_scheme import avalanche_model, get_slope
from models.shallow_water_solver import shallow_water_solver
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import scipy
import scipy.ndimage as sciim
from scipy.signal import find_peaks       
        
"""
    This is the base class for the model 
"""
class NullSimpleHydroMorphologicalModel(object):
    """

    """

    def __init__(self):
        self._useAvalanche = True
        self._useSmoother = False
        self._verts = []
        self._tsteps = []
        self._h = None
        self._q = None
        self._u = None
        
    def setup_bed_properties(self, D50, repose_angle = 30., rho_particle = 2650., nP = 0.4):
        self._D50 = D50
        self._rho_particule = rho_particle
        self._repose_angle = repose_angle
        self._threshold_angle = repose_angle + 1.0
        self._adjustment_angle = repose_angle - 2.0
        self._nP = nP
            
    def configure_avalanche_model(self, threshold_angle, repose_angle, adjustment_angle):
        self._threshold_angle = threshold_angle
        self._repose_angle = repose_angle
        self._adjustment_angle = adjustment_angle
        
    def flow_boundary_conditions(self, qin, surface):
        self._qin = qin
        self._surface = surface
        
    def setup_domain(self, xc, zc, dx):
        self._xc = xc
        self._zc = zc
        self.z0 = zc.copy()
        self._nx = len(xc)
        self._dx = dx
        
        

    def setup_morpho_model(self, exner_model, 
                           useAvalanche = True, 
                           useSmoother = True, 
                           a=0.001, 
                           b = 3.0):
        
        self._useAvalanche = useAvalanche
        self._useSmoother = useSmoother
        self._a = a
        self._b = b
        self._exner_model = exner_model

        
    
    def _apply_smoothing_filter(self, x, z):
        # ----------------------------------
        # Apply the two-step smooting scheme Eq. 6 in Niemann et al 2011.
        # ----------------------------------
        
        nx = len(z)
        zhat = np.zeros(nx)
        zsmooth = np.zeros(nx)
        for i in range(0, nx):  # i=2
            zlocal = get_stencil(z, i - 1, i + 2)
            zhat[i] = 0.5*zlocal[1] + 0.25*(zlocal[0]+zlocal[2])

        for i in range(0, nx):
            zhatlocal = get_stencil(zhat, i - 1, i + 2)
            zsmooth = (3./2.)*zhatlocal[1] - 0.25*(zhatlocal[0]+zhatlocal[2])
        return zsmooth
            
    def get_peaks(self, z):
        top_peaks, _ = find_peaks(z, height = z.mean()*1.1)
        bottom_peaks, _ = find_peaks(-1.*z, height = z.mean()*-0.8, distance = int(1.2/dx))

        '''ztop = [z[i] for i in top_peaks]
        zbottom = [z[i] for i in bottom_peaks]

        ztop = np.array(ztop)
        zbottom = np.array(zbottom)'''
        
        return top_peaks, bottom_peaks
            
    def _calculate_bedload(self, x, z):
        top_peaks, bottom_peaks = get_peaks(z)
        xcrests = [x[i] for i in top_peaks]
        zcrests = [z[i] for i in top_peaks]
        
        xbottom = [x[i] for i in bottom_peaks]
        zbottom = [z[i] for i in bottom_peaks]
        
        xreattachments = []
        
        for i in range(len(top_peaks)):
        
            if top_peaks[i] > bottom_peaks[i]:
                # At the top
                height = zcrests[i] - zbottom[i]
                xreattachment = (5.* height) + xcrests[i]
                
                index = bottom_peaks[i]
                while x[index] < xreattachment:
                    index += 1
                
                xreattachments.append([index])
                
            else:
                print('not implemented')
            
        
        
        
        qbedload = np.zeros(self._nx)
        for i in range(0,self._nx): #i=2 
            a = self._a
            b = self._b
            qin = self._qin
            surface = self._surface
            u = qin/(surface-z[i])
            qbedload[i] = (a *u**b)
            
        return qbedload
        
    
    def _avalanche_model(self, x, z):
        # Apply the avalanche model
        dx = x[1] - x[0]
        znew, iterations1 = avalanche_model(dx, x, z, max_iterations = 100, 
                                          threshold_angle = self._threshold_angle, 
                                          angle_of_repose = self._repose_angle, 
                                          adjustment_angle = self._adjustment_angle)
        
        # Now flip it to run in reverse
        zflip = np.flip(znew, axis=0)
        zflip, iterations1 = avalanche_model(dx, x, zflip, max_iterations = 100, 
                                          threshold_angle = self._threshold_angle, 
                                          angle_of_repose = self._repose_angle, 
                                          adjustment_angle = self._adjustment_angle)
        
        znew = np.flip(zflip, axis = 0)
        return znew
    
    
    def _extract_results(self, x, z, qbedload, timestep, dt, fileName):
        self._verts.append(list(zip(self._xc.copy(),self._zc.copy(), qbedload.copy())))
        self._tsteps.append(timestep)
                
        if fileName != None:
            np.save(fileName, verts)

        
        #print('Time step: {0} mins - uavg: {1} - Elevation {2}'.format(timestep/60., u.mean(), surf.mean()))
        #print('Courant number: {0}'.format(courant))
                    
    
    def run(self, simulationTime, dt, extractionTime, fileName):
        pass



class SimpleHydroMorphologicalModel(NullSimpleHydroMorphologicalModel):

    
    def run(self, simulationTime, dt, extractionTime, fileName):

        print(' Starting simulation....')
        # --------------------------------
        #  Setup the model run parameters
        # --------------------------------
        nt = int(simulationTime / dt)  # Number of time steps
        print('Number of time steps: {0} mins'.format(nt/60.))
        slope = np.gradient(self._zc, self._dx)

        # --------------------------------
        # Set up the domain, BCs and ICs
        # --------------------------------
        print('Grid dx = {0}'.format(self._dx))
        print('Grid nx = {0}'.format(self._nx))
        

        
        print('Completed the intialization of the model')
        print('D50:    {0}'.format(self._D50))
        print('Rho Particle:    {0}'.format(self._rho_particule))
        print('Angle Repose Degrees:    {0}'.format(self._repose_angle))
        # --------------------------------
        # Initialize the sed transport
        # --------------------------------
        print('Zc = {0}'.format(len(self._zc)))
        qbedload = self._calculate_bedload(self._zc)
        print('Max qbedload = {0}'.format(qbedload.max()))
        

        # --------------------------------
        #  Run the model
        # --------------------------------
        cntr = 0
        for n in range(1, nt):
            
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
            else:
                print('Warning - not using avalanche')
            
            # --------------------------------
            # Appy smoothing filter
            # --------------------------------
            if self._useSmoother == True:
                znp1 = self._apply_smoothing_filter(self._xc, znp1)
                print('Applying smoothing')
            
            # --------------------------------
            # update the bedload 
            # --------------------------------
            slope = np.gradient(znp1,self._dx)
            qbedload = self._calculate_bedload(znp1)
            
            self._zc = znp1
            
            
            if (n*dt / extractionTime) == math.floor(n*dt / extractionTime):        
                timestep = n*dt
                self._extract_results(self._xc, self._zc, qbedload, timestep, dt, fileName)              

        return self._zc, qbedload
            
            
        
            
