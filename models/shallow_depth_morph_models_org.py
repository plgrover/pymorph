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
from scipy.interpolate import interp1d

class NullExnerModel(object):
    def update_bed(self, zc, qbedload, dt, baseModel):
        pass


    
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

        for i in range(buffer, baseModel._nx-buffer): #i=2      
            qloc = weno.get_stencil(qbedload,i-1,i+2)  
            zhatn[i] = z[i]-(1./(1.- baseModel._nP))*dt/(baseModel._dx)*(qloc[1] - qloc[0])

        # Now update hydraulics using z1, and update the bedload
        h1, u1, q1 = baseModel._update_hydrodynamic_model(baseModel._h, baseModel._q, baseModel._xc, zhatn, baseModel._update_time)
        #slope1 = np.gradient(z1)
        qbedload1 = baseModel._calculate_bedload(h1, u1, baseModel._xc, zhatn, baseModel._a, baseModel._b)

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
            
        # Now update hydraulics using z1, and update the bedload
        h1, u1, q1 = baseModel._update_hydrodynamic_model(baseModel._h, baseModel._q, baseModel._xc, z1, baseModel._update_time)
        #slope1 = np.gradient(z1)
        
        # CHECK THIS!!!!!!!
        slope = np.gradient(z1,baseModel._dx)
        qbedload = baseModel._calculate_bedload(h1, u1, slope)
        
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
class NullShallowHydroMorphologicalModel(object):
    """
    """

    def __init__(self):
        self._useAvalanche = True
        self._useSmoother = False
        self._sed_model = 'bagnold'
        self._verts = []
        self._tsteps = []
        self._h = None
        self._q = None
        self._u = None
        self._mannings = None
        self._ks = None
        self._ycr_factor = 1.0
        self._modify_bedload_flag = False
        self._periodic_reattachment_flag = False
        self._time = 0.0
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
        
    def flow_boundary_conditions(self, qin, sOut):
        self._qin = qin
        self._sOut = sOut
        
    def setup_domain(self, xc, zc, dx):
        self._xc = xc
        self._zc = zc
        self.z0 = zc.copy()
        self._nx = len(xc)
        self._dx = dx
        
        # Find the location of the first peak
        peaks, _ = find_peaks(zc, height=0.02)
        self._peak1 = peaks[0]
        self._wave_speed = {}
        self._wave_length = {}
        self._wave_height = {}

    def setup_morpho_model(self, exner_model, 
                           useAvalanche = True, 
                           useSmoother = True, 
                           sed_model='bagnold', 
                           useSlopeAdjust = False):
        
        self._useAvalanche = useAvalanche
        self._useSmoother = useSmoother
        self._sed_model = sed_model
        self._useSlopeAdjust = useSlopeAdjust
        self._exner_model = exner_model
        
    def setup_mannings_hydro_model(self, mannings, bed_slope):
        self._mannings = mannings
        self._ks = 0.0033
        self._bed_slope = bed_slope
        self._sws = None
        self._update_time = 10
        
    def setup_chezy_hydro_model(self, ks, bed_slope):
        self._ks = ks
        self._bed_slope = bed_slope
        self._sws = None
        self._update_time = 10
        
    def _init_hydrodynamic_model(self, tfinal=300., max_steps=100000):
        #--------------------------------
        # Initalize the model
        #--------------------------------
        self._sws = shallow_water_solver(kernel_language='Fortran')
        self._sws.set_solver(max_steps=max_steps)
        self._sws.set_state_domain(self._xc, self._zc)
        
        if self._mannings == None:
            self._sws.set_chezy_source_term(ks = self._ks, slope = self._bed_slope)
        else:
            self._sws.set_mannings_source_term(mannings=self._mannings, slope=self._bed_slope)
        
        self._sws.set_Dirichlet_BC(self._sOut, self._qin)
        self._sws.set_inital_conditions(self._sOut, 0.0)
        self._sws.set_controller(tfinal = tfinal, num_output_times=1)
        self._sws.run()
        
        h = self._sws.get_hf()
        u = self._sws.get_uf()
        q = self._sws.get_qf()

        return h, u, q
    
    def _adjust_ycr(self, factor):
        self._ycr_factor = factor
        
    
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
        self._sws = shallow_water_solver(kernel_language='Fortran')
        self._sws.set_solver(max_steps=max_steps)
        self._sws.set_state_domain(x, z)
        if self._mannings == None:
            self._sws.set_chezy_source_term(ks = self._ks, slope = self._bed_slope)
        else:
            self._sws.set_mannings_source_term(mannings=self._mannings, slope=self._bed_slope)
        self._sws.set_Dirichlet_BC(self._sOut, self._qin)
        self._sws.set_conditions_from_previous(h, q)
        self._sws.set_controller(tfinal = tfinal, num_output_times = 1)
        self._sws.run()
        
        h = self._sws.get_hf()
        u = self._sws.get_uf()
        q = self._sws.get_qf()

        return h, u, q
    
    def _calculate_wave_height(self, z, timestep):
        top_peaks, _ = find_peaks(z, height = 0.02)
        bottom_peaks, _ = find_peaks(-1.*z, height = -0.02)
        
        ztop = [z[i] for i in top_peaks]
        zbottom = [z[i] for i in bottom_peaks]
        
        ztop = np.array(ztop)
        zbottom = np.array(zbottom)
        
        self._wave_height[timestep] = ztop.mean() - zbottom.mean()
                
            
    
    def _calculate_wave_length(self, z, timestep):
        peaks, _ = find_peaks(-1.*z, height = -0.02)
        lengths = []
        last_peak = None
        for peak in peaks:
            if last_peak == None:
                last_peak = peak
            else:
                lengths.append(self._dx * (peak-last_peak))
                
        lengths = np.array(lengths)
        self._wave_length[timestep] = lengths.mean()
    
    def _calculate_wave_speed(self, z, timestep):
        # Note that we are picking off the bttom part 
        # of the dune.
        peaks, _ = find_peaks(-1.*z, height = -0.02)
        peak_new = None
        for peak in peaks:
            if peak > self._peak1:
                peak_new = peak
                break
        distance = (peak_new - self._peak1) * self._dx
        
        self._wave_speed[timestep] = distance
            
    def _calculate_bedload(self, h, u, slope):
        qbedload = np.zeros(self._nx)
        D50 = self._D50 * self._ycr_factor
        
        for i in range(0,self._nx):
            qbedload[i] = sedtrans.get_unit_bed_load_slope(h[i], u[i], D50, slope[i], 
                                                       self._rho_particule, 
                                                       angleReposeDegrees = self._repose_angle, 
                                                       type=self._sed_model,
                                                        useSlopeAdjust= self._useSlopeAdjust)
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
    
    
    def _extract_results(self, x, z, u, q, h, qbedload, timestep, dt, fileName):
        self._verts.append(list(zip(self._xc.copy(),self._zc.copy(), u.copy(), q.copy(), h.copy(), qbedload.copy())))
        self._tsteps.append(timestep)
                
        if fileName != None:
            np.save(fileName, verts)
            
        courants = weno.get_Phase_Speeds(qbedload, z, self._nP)*dt/self._dx
        surf = self._zc + h
        
        print('Time step: {0} mins - uavg: {1} - Elevation {2}'.format(timestep/60., u.mean(), surf.mean()))
        print('Courant number: max {0}, mean{1}'.format(courants.max(), courants.mean()))
                    
    
    def run(self, simulationTime, dt, extractionTime, fileName):
        pass

    def get_wave_dataframe(self):
        wavehdf = pd.DataFrame.from_dict(self._wave_height, orient='index',columns=['height'])

        waveldf = pd.DataFrame.from_dict(self._wave_length, orient='index',columns=['length'])

        wavesdf = pd.DataFrame.from_dict(self._wave_speed, orient='index',columns=['speed'])

        waveDf = pd.concat([wavehdf, waveldf, wavesdf], axis=1)
        
        return waveDf

class ShallowHydroMorphologicalModel(NullShallowHydroMorphologicalModel):

    
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
        
        # --------------------------------
        # Initialize the hydro model transport
        # --------------------------------
        print('Initializing hydrodynamic model...')
        h, u, q = self._init_hydrodynamic_model()
        self._h  = h
        self._u = u
        self._q = q
        print('Completed the intialization of the model')

        print('D50:    {0}'.format(self._D50))
        print('Rho Particle:    {0}'.format(self._rho_particule))
        print('Angle Repose Degrees:    {0}'.format(self._repose_angle))
        # --------------------------------
        # Initialize the sed transport
        # --------------------------------
        qbedload = self._calculate_bedload(h, u, slope)       
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
            # update the flow
            # --------------------------------
            
            if cntr > -1:
                h, u, q = self._update_hydrodynamic_model(h, q, 
                                                      self._xc, znp1, tfinal=self._update_time)
                self._h  = h
                self._u = u
                self._q = q
                cntr = 0
            cntr += 1
            
            # --------------------------------
            # update the bedload 
            # --------------------------------
            slope = np.gradient(znp1,self._dx)
            qbedload = self._calculate_bedload(self._h, self._u, slope)
            
            self._zc = znp1
            
            
            if (n*dt / extractionTime) == math.floor(n*dt / extractionTime):        
                timestep = n*dt
                self._extract_results(self._xc, self._zc, u, q, h, qbedload, timestep, dt, fileName)              
                #self._calculate_wave_speed(self._zc, timestep)
                #self._calculate_wave_length(self._zc, timestep)
                #self._calculate_wave_height(self._zc, timestep)
        return self._zc, u, q, h, qbedload

    
class ModifiedShallowHydroMorphologicalModel(NullShallowHydroMorphologicalModel):

    def _get_top_peaks(self, z, dx):
        top_peaks, _ = find_peaks(z, height = z.mean()*1.1, distance = int(0.5/dx))
        return top_peaks

    def _get_bottom_indexes(self, z, crest_indexes, dx):
        bottom_indexes = []

        for crest_index in crest_indexes:

            index_end = crest_index + int(0.5*1.3/dx) 
            minZ = np.amin(z[crest_index : index_end])
            lowPoints = np.where(z == minZ)[0]

            index = np.where(lowPoints > crest_index)[0][0]
            index = lowPoints[index]
            bottom_indexes.append(index)

        return np.array(bottom_indexes)



    def _calculate_bedload(self, h, u, x, z, a, b):
        
        if a is None:
            a = self._a
        if b is None:
            b = self._b
        
        qbedload = np.zeros(len(x))
        # Match 3d
        #a = 0.00003
        #b = 2.
        
        #match 2d
        #a = 0.00005
        #b = 5
        qbedload = (a*u*np.abs(u) **(b-1.))
        if self._modify_bedload_flag == True:
            qbedload = self._modify_bedload(qbedload, x, z, scale_factor=self._scale_factor)
            #print('Modified the bedload with factor {0}'.format(self._scale_factor))
            
        return qbedload



    def _get_recirculation_indexes(self, x, z, crest_indexes, bottom_indexes):

        recirculation_indexes = []
        for i in range(len(crest_indexes)):

            height = z[crest_indexes[i]] - z[bottom_indexes[i]]
            
            reattachment_pos = 5.
            if self._periodic_reattachment_flag == True:
                fxr = 0.7
                #reattachment_pos = reattachment_pos + np.sin(2.*math.pi*fxr*self._time)
                reattachment_pos = reattachment_pos + np.sin(2*math.pi*fxr*self._time)
                #print(reattachment_pos, self._time)
            
            xreattachment = (reattachment_pos * height) + x[crest_indexes[i]]

            index = bottom_indexes[i]
            while index < len(x) and x[index] < xreattachment:
                index += 1

            recirculation_indexes.append(index)

        return np.array(recirculation_indexes)

    def _modify_bedload(self, qb, x, z, scale_factor):
        qb_new = qb.copy()
        
        dx = x[1] - x[0]
        crest_indexes = self._get_top_peaks(z, dx)
        base_indexes = self._get_bottom_indexes(z, crest_indexes, dx)
        reattachment_indexes = self._get_recirculation_indexes(x, z, crest_indexes, base_indexes)
        
        overshoot = 2
        
        # Adjust as required
        for i in range(len(crest_indexes)):

            qsb_reattachment = qb[reattachment_indexes[i]]

            if i+1 < len(crest_indexes):
                for j in range(crest_indexes[i] + overshoot, reattachment_indexes[i] ):
                    qb_new[j] =  0 # qb[j] - qsb_reattachment

                '''for j in range(reattachment_indexes[i], crest_indexes[i+1]):
                    qb_new[j] = qb[j] # *(x[j] - x[reattachment_indexes[i]] )**scale_factor/(x[crest_indexes[i+1]] - x[reattachment_indexes[i]])**scale_factor'''

        return qb_new

    def set_scale_factor(self, scale_factor):
        self._scale_factor = scale_factor
    
    def use_modifier(self):
        self._modify_bedload_flag = True
        
    def use_periodic_reattachment(self):
        self._periodic_reattachment_flag = True
        
    def set_grass_parameters(self, a, b):
        self._a = a
        self._b = b
    
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
        
        # --------------------------------
        # Initialize the hydro model transport
        # --------------------------------
        print('Initializing hydrodynamic model...')
        h, u, q = self._init_hydrodynamic_model()
        self._h  = h
        self._u = u
        self._q = q
        print('Completed the intialization of the model')

        print('D50:    {0}'.format(self._D50))
        print('Rho Particle:    {0}'.format(self._rho_particule))
        print('Angle Repose Degrees:    {0}'.format(self._repose_angle))
        # --------------------------------
        # Initialize the sed transport
        # --------------------------------
        qbedload = self._calculate_bedload(h, u, self._xc, self._zc, self._a, self._b) 
      

        # --------------------------------
        #  Run the model
        # --------------------------------
        cntr = 0
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
            
            # --------------------------------
            # update the flow
            # --------------------------------
            
            if cntr > -1:
                h, u, q = self._update_hydrodynamic_model(h, q, 
                                                      self._xc, znp1, tfinal=self._update_time)
                self._h  = h
                self._u = u
                self._q = q
                cntr = 0
            cntr += 1
            
            # --------------------------------
            # update the bedload 
            # --------------------------------
            qbedload = self._calculate_bedload(self._h, self._u, self._xc, znp1, self._a, self._b)
            
            self._zc = znp1
            
            
            if (time / extractionTime) == math.floor(time / extractionTime):        
                timestep = n*dt
                self._extract_results(self._xc, self._zc, u, q, h, qbedload, timestep, dt, fileName)              
                #self._calculate_wave_speed(self._zc, timestep)
                #self._calculate_wave_length(self._zc, timestep)
                #self._calculate_wave_height(self._zc, timestep)
        return self._zc, u, q, h, qbedload

    
class ParameterizedMorphologicalModel(NullShallowHydroMorphologicalModel):

    def set_bedload_model(self, bedloadModel):
        self._bedloadModel = bedloadModel


    def _calculate_bedload(self, h, u, xc, z, a, b):
        self._bedloadModel.calculate_bedload(h, u, xc, z, self._time)

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

        qbedload = [((zs - self.__z_offset)/self.__delta * self.__qsb_max) for zs in z]
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



    
class NullQuasiSteadyExnerModel(object):
    def update_bed(self, zc, qb, qbstar, dt, baseModel):
        pass
    
class EulerQuasiSteadyExnerModel(NullQuasiSteadyExnerModel):
    
    def update_bed(self, zn, qbedload, dt, L, baseModel):
        buffer = 0
        
        znp1 = np.zeros(baseModel._nx)
        
        for i in range(buffer, baseModel._nx-buffer): #i=2
            floc = weno.get_stencil(flux, i - 1, i + 1)
            znp1[i] = zn[i] - (dt/(1. - baseModel._nP))*(1./L)*(qb[i] - qbstar[i])
            
        return znp1
    
    
class ShallowUnsteadyHydroMorphologicalModel(NullShallowHydroMorphologicalModel):
    
    def _get_qb(self, qbstar, L):
        buffer = 0
        tol = 1.e-16
        dt = 0.0001
        qb =np.zeros(self._nx) 
        dx = self._dx
        
        for iter in range(500):
            qbnew = np.zeros(self._nx) 
            
            for i in range(buffer, self._nx - buffer):  
                qloc = weno.get_stencil(qb,i-1,i+2) 
                
                qbnew[i] = qloc[1] - (dt/(2.*dx))*(qloc[2] + qb[0]) + (dt/L)*(qbstar[i]-qloc[1])
                
                #qbnew[i+1]= qbstar[i+1] +(qb[i] - qbstar[i])*math.exp(-dx/(2.*L)) + (qbstar[i] - qbstar[i+1])*(2*L/dx)*(1-math.exp(-dx/(2.*L)))
            
            resid = np.mean(qbnew-qb) 
            qb = qbnew.copy()
            if resid < tol:
                print('break')
                break
        print(resid)
        return qb
    
    def set_grass_parameters(self, a, b):
        self._a = a
        self._b = b
    
    def _calculate_bedload(self, h, u, slope):
        qbedload = np.zeros(self._nx)
        
        
        #match 2d
        a = 0.00005
        b = 5
        qbedload = (a*u*np.abs(u) **(b-1.))
        
        ''' D50 = self._D50 * self._ycr_factor
        print('calculating')
        for i in range(0,self._nx):
            qbedload[i] = sedtrans.get_unit_bed_load_slope(h[i], u[i], D50, slope[i], 
                                                       self._rho_particule, 
                                                       angleReposeDegrees = self._repose_angle, 
                                                       type=self._sed_model,
                                                        useSlopeAdjust= self._useSlopeAdjust)'''
        return qbedload
    
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
        
        # --------------------------------
        # Initialize the hydro model transport
        # --------------------------------
        print('Initializing hydrodynamic model...')
        h, u, q = self._init_hydrodynamic_model()
        self._h  = h
        self._u = u
        self._q = q
        print('Completed the intialization of the model')

        print('D50:    {0}'.format(self._D50))
        print('Rho Particle:    {0}'.format(self._rho_particule))
        print('Angle Repose Degrees:    {0}'.format(self._repose_angle))
        # --------------------------------
        # Initialize the sed transport
        # --------------------------------
        qbstar = self._calculate_bedload(h, u, slope)    
        qb = self._get_qb( qbstar, L)
        
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
            # update the flow
            # --------------------------------
            
            if cntr > -1:
                h, u, q = self._update_hydrodynamic_model(h, q, 
                                                      self._xc, znp1, tfinal=self._update_time)
                self._h  = h
                self._u = u
                self._q = q
                cntr = 0
            cntr += 1
            
            # --------------------------------
            # update the bedload 
            # --------------------------------
            slope = np.gradient(znp1,self._dx)
            qbedload = self._calculate_bedload(self._h, self._u, slope)
            
            self._zc = znp1
            
            
            if (n*dt / extractionTime) == math.floor(n*dt / extractionTime):        
                timestep = n*dt
                self._extract_results(self._xc, self._zc, u, q, h, qbedload, timestep, dt, fileName)              
                #self._calculate_wave_speed(self._zc, timestep)
                #self._calculate_wave_length(self._zc, timestep)
                #self._calculate_wave_height(self._zc, timestep)
        return self._zc, u, q, h, qbedload