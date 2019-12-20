#! /usr/bin/env python
# -*- coding: utf-8 -*

""" 
    This module contains the simple hydrodynamic model (conservation of mass) and morphodyamic models.
"""

import math
import schemes.weno as weno
import sediment_transport.sed_trans as sedtrans
from schemes.avalanche_scheme import avalanche_model, get_slope
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import scipy
import scipy.ndimage as sciim
from scipy.signal import find_peaks


class NullExnerModel(object):
    def update_bed(self, zc, qbedload, dt, baseModel):
        pass

class EulerUpwindModel(NullExnerModel):
    def update_bed(self, zn, qbedload, dt, baseModel, buffer = 10):
        znp1 = np.zeros(baseModel._nx)
        for i in range(buffer, baseModel._nx-buffer): #i=2       
            qloc = weno.get_stencil(qbedload,i-2,i+4) 
            zloc = weno.get_stencil(zn,i-2,i+4)
            
            # Determine the Upwind flux
            # The 0.5 comes from the c+abs(c) which is 2 if the wave speed is +ive
            # this is the evaluation of the left and right based fluxes. Eq. 18 and 19
            alpha = 0.
            if (zloc[3]-zloc[2]) == 0.0:
                alpha = np.sign( (qloc[3]-qloc[2]) )
            else:
                alpha = np.sign( (qloc[3]-qloc[2])/ (zloc[3]-zloc[2]) )
                
            qloc = weno.get_stencil(qbedload,i-1,i+2)  
            znp1[i] = zn[i]-(1./(1. - baseModel._nP))*(dt/baseModel._dx)*0.5*( (1 + alpha)*(qloc[1] - qloc[0])  + (1 - alpha)*(qloc[2] - qloc[1]))
            
        return znp1
    
class EulerCentredModel(NullExnerModel):
    def update_bed(self, z, qbedload, dt, baseModel, buffer = 10):
        znp1 = np.zeros(baseModel._nx)
        for i in range(buffer, baseModel._nx-buffer): #i=2       
            qloc = weno.get_stencil(qbedload,i-1,i+2)  
            znp1[i] = z[i]-(1./(1.-nP))*dt/(baseModel._dx*2.)*(qloc[2] - qloc[0])
        return znp1

class MacCormackModel(NullExnerModel):
    
    def update_bed(self, z, qbedload, dt, baseModel, buffer = 0):
        znp1 = np.zeros(baseModel._nx)
        zhatn = np.zeros(baseModel._nx)
        
        #print('Hey dude',len(qbedload))

        for i in range(buffer, baseModel._nx-buffer): #i=2      
            qloc = weno.get_stencil(qbedload,i-1,i+2)  
            zhatn[i] = z[i]-(1./(1.- baseModel._nP))*dt/(baseModel._dx)*(qloc[1] - qloc[0])

        #slope1 = np.gradient(z1)
        qbedload1 = baseModel._calculate_bedload(zhatn)
        
        for i in range(buffer, baseModel._nx-buffer): #i=2       
            qloc = weno.get_stencil(qbedload1,i - 1, i + 2)  
            znp1[i] = 0.5*(zhatn[i]+z[i]) - (1/(1.- baseModel._nP))*dt/(baseModel._dx*2.)*(qloc[2] - qloc[1])

        return znp1        

class EulerWenoModel(NullExnerModel):
    
    def update_bed(self, z, qbedload, dt, baseModel, buffer = 0):
        
        znp1 = np.zeros(baseModel._nx)
        flux = np.zeros(baseModel._nx)
        for i in range(buffer, baseModel._nx-buffer): #i=2
            zloc = weno.get_stencil(z, i-2, i+4)        
            # Since k=3
            # stencil is i-2 to i+2 
            qloc = weno.get_stencil(qbedload, i-2, i+4)
            
            # Determine the Upwind flux
            # The 0.5 comes from the c+abs(c) which is 2 if the wave speed is +ive
            # this is the evaluation of the left and right based fluxes. Eq. 18 and 19        
            roe_speed = 0.0
            if (zloc[3]-zloc[2]) == 0.0:
                roe_speed = np.sign( (qloc[3] - qloc[2]) )
            else:
                roe_speed = np.sign( (qloc[3] - qloc[2])/(zloc[3] - zloc[2]) )

            if roe_speed >= 0.0:
                flux[i] = weno.get_left_flux(qloc)
            else:
                flux[i] = weno.get_right_flux(qloc)
                
        # Need the sign of the phase speed
        # Need to check this out
        for i in range(buffer, baseModel._nx-buffer): #i=2
            floc = weno.get_stencil(flux, i - 1, i + 1)
            znp1[i] = z[i] - (1./(1. - baseModel._nP))*(dt/(baseModel._dx))*(floc[1] - floc[0])
            
        return znp1

    
class TVD2ndWenoModel(NullExnerModel):
    
    def update_bed(self, z, qbedload, dt, baseModel, buffer = 0):
        
        # Step 1 - Estimate z1
        z1 = np.zeros(baseModel._nx) 
        znp1 = np.zeros(baseModel._nx)
        flux = np.zeros(baseModel._nx)
        
        # Note that I have modified these to move away from the inlet/outlet
        for i in range(buffer, baseModel._nx-buffer): #i=2
            zloc = weno.get_stencil(z, i-2, i+4)        
            # Since k=3
            # stencil is i-2 to i+2 
            qloc = weno.get_stencil(qbedload, i-2, i+4)
            
            # Determine the Upwind flux
            # The 0.5 comes from the c+abs(c) which is 2 if the wave speed is +ive
            # this is the evaluation of the left and right based fluxes. Eq. 18 and 19        
            roe_speed = 0.0
            if (zloc[3]-zloc[2]) == 0.0:
                roe_speed = np.sign( (qloc[3] - qloc[2]) )
            else:
                roe_speed = np.sign( (qloc[3] - qloc[2])/(zloc[3] - zloc[2]) )

            if roe_speed >= 0.0:
                flux[i] = weno.get_left_flux(qloc)
            else:
                flux[i] = weno.get_right_flux(qloc)
                
        for i in range(buffer, baseModel._nx-buffer): #i=2
            floc = weno.get_stencil(flux,i - 1, i + 1)
            z1[i] = z[i]-(1./(1.-baseModel._nP))*dt/baseModel._dx*(floc[1] - floc[0])
            
        #slope1 = np.gradient(z1)
        qbedload1 = baseModel._calculate_bedload(z1)
        
        for i in range(buffer, baseModel._nx-buffer): #i=2
            # Now update based on the updated values
            zloc = weno.get_stencil(z1, i-2, i+4)        
            # Since k=3
            # stencil is i-2 to i+2 
            qloc = weno.get_stencil(qbedload, i-2, i+4)
            
            # Determine the Upwind flux
            # The 0.5 comes from the c+abs(c) which is 2 if the wave speed is +ive
            # this is the evaluation of the left and right based fluxes. Eq. 18 and 19        
            roe_speed = 0.0
            if (zloc[3]-zloc[2]) == 0.0:
                roe_speed = np.sign( (qloc[3] - qloc[2]) )
            else:
                roe_speed = np.sign( (qloc[3] - qloc[2])/(zloc[3] - zloc[2]) )

            if roe_speed >= 0.0:
                flux[i] = weno.get_left_flux(qloc)
            else:
                flux[i] = weno.get_right_flux(qloc)
                
        for i in range(buffer, baseModel._nx-buffer): #i=2       
            floc = weno.get_stencil(flux,i - 1, i + 1)
            znp1[i] = 0.5*z[i] + 0.5*z1[i] - 0.5*(1./(1.-baseModel._nP))*dt/baseModel._dx*(floc[1] - floc[0])
        
        return znp1

        
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
        self._update_time = 10
        
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
            zlocal = weno.get_stencil(z, i - 1, i + 2)
            zhat[i] = 0.5*zlocal[1] + 0.25*(zlocal[0]+zlocal[2])

        for i in range(0, nx):
            zhatlocal = weno.get_stencil(zhat, i - 1, i + 2)
            zsmooth[i] = (3./2.)*zhatlocal[1] - 0.25*(zhatlocal[0]+zhatlocal[2])
        return zsmooth
            
        
    def _update_hydrodynamic_model(self, h, q, x, z, tfinal=10., max_steps=100000):
        pass
            
    def _calculate_bedload(self, z):
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
            
            # --------------------------------
            # Appy smoothing filter
            # --------------------------------
            if self._useSmoother == True:
                znp1 = self._apply_smoothing_filter(self._xc, znp1)
            
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
            