#! /usr/bin/env python
# -*- coding: utf-8 -*

""" 
    This module contains the shallow-water hydrodynamic model and morphodyamic models.
"""

import os,sys

import math
import collections
import numpy as np

sys.path.append("/pymorph")
from schemes.weno import get_left_flux,get_right_flux,get_Max_Phase_Speed
from schemes.weno import get_stencil
import sediment_transport.sed_trans as sedtrans
import morph_geom_lib as mgl
from schemes.avalanche_scheme import *
from models.exner_models import *
        
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
    
    def get_h(self):
        return self._sws.get_hf()
    
    def get_u(self):
        return self._sws.get_hf()
    
    def get_q(self):
        return self._sws.get_qf()
    
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

'''
This model 
'''
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
