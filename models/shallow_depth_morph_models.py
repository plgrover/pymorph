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
from scipy.signal import savgol_filter
import scipy
import scipy.ndimage as sciim


class NullExnerModel(object):
    def update_bed(self, zc, qbedload, dt, baseModel):
        pass
    
    
class EulerWenoModel(NullExnerModel):
    
    def update_bed(self, z, qbedload, dt, baseModel):
        
        znp1 = np.zeros(baseModel._nx)
        flux = np.zeros(baseModel._nx)
        for i in range(0, baseModel._nx): #i=2
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
        for i in range(0, baseModel._nx): #i=2       
            floc = weno.get_stencil(flux,i - 1, i + 1)
            znp1[i] = z[i]-(1./(1.-baseModel._nP))*dt/baseModel._dx*(floc[1] - floc[0])
            
        return znp1


"""
    This is the base class for the model 
"""
class NullShallowHydroMorphologicalModel(object):
    """

    """

    def __init__(self):
        self._type = 'bagnold'
        self._avalanche = True
        self._useSmoother = False
        self._sed_model = 'bagnold'
        self._verts = []
        self._tsteps = []

        
    def setup_bed_properties(self, D50, repose_angle = 30., rho_particle = 2650., nP = 0.4):
        self._D50 = D50
        self._rho_particule = rho_particle
        self._repose_angle = repose_angle
        self._threshold_angle = repose_angle + 0.5
        self._adjustment_angle = repose_angle - 0.5
        self._nP=nP
            
    def configure_avalanche_model(self, threshold_angle, repose_angle, adjustment_angle):
        self._threshold_angle = threshold_angle
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
        
    def setup_hydro_model(self, mannings, slope):
        self._mannings = mannings
        self._bed_slope = slope
        self._sws = None
        
    def _init_hydrodynamic_model(self, tfinal=300., max_steps=100000):
        #--------------------------------
        # Initalize the model
        #--------------------------------
        self._sws = shallow_water_solver(kernel_language='Fortran')
        self._sws.set_solver(max_steps=max_steps)
        self._sws.set_state_domain(self._xc, self._zc)
        self._sws.set_mannings_source_term(mannings=self._mannings, slope=self._bed_slope)
        self._sws.set_Dirichlet_BC(self._sOut, self._qin)
        self._sws.set_inital_conditions(self._sOut, 0.0)
        self._sws.set_controller(tfinal = tfinal, num_output_times=1)
        self._sws.run()
        
        h = self._sws.get_hf()
        u = self._sws.get_uf()
        q = self._sws.get_qf()

        return h, u, q
        
        

    def _update_hydrodynamic_model(self, h, q,  tfinal=10., max_steps=100000):
        self._sws = shallow_water_solver(kernel_language='Fortran')
        self._sws.set_solver(max_steps=max_steps)
        self._sws.set_state_domain(self._xc, self._zc)
        self._sws.set_mannings_source_term(mannings=self._mannings, slope=self._bed_slope)
        #print('----------------------------')
        #print(self._sOut, self._qin)
        self._sws.set_Dirichlet_BC(self._sOut, self._qin)
        self._sws.set_conditions_from_previous(h, q)
        self._sws.set_controller(tfinal = tfinal, num_output_times = 1)
        self._sws.run()
        
        h = self._sws.get_hf()
        u = self._sws.get_uf()
        q = self._sws.get_qf()

        return h, u, q
        
    def _calculate_bedload(self, h, u, slope):
        qbedload = np.zeros(self._nx)
        
        # Nov 13 2018 - Can modify later
        for i in range(0,self._nx):
            qbedload[i] = sedtrans.get_unit_bed_load_slope(h[i], u[i], self._D50, slope[i], 
                                                       self._rho_particule, 
                                                       angleReposeDegrees = self._repose_angle, 
                                                       type=self._sed_model,
                                                        useSlopeAdjust=self._useSlopeAdjust)
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
    
    def run(self, simulationTime, dt, extractionTime, fileName):
        pass



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
        for n in range(1, nt):

            slope = np.gradient(self._zc, self._dx)
            # --------------------------------
            # Update the bed
            # --------------------------------
            self._zc = self._exner_model.update_bed(self._zc, qbedload, dt, self)
            
            # --------------------------------
            # Avalanche the bed
            # --------------------------------
            self._zc = self._avalanche_model(self._xc, self._zc)
            
            # --------------------------------
            # Appy smoothing filter
            # --------------------------------
            
            # --------------------------------
            # update the flow
            # --------------------------------
            
            h, u, q = self._update_hydrodynamic_model(h, q, tfinal=10.)
            
            # --------------------------------
            # update the bedload 
            # --------------------------------
            slope = np.gradient(self._zc,self._dx)
            qbedload = self._calculate_bedload(h, u, slope)
            
            
            if (n*dt / extractionTime) == math.floor(n*dt / extractionTime):
                self._verts.append(list(zip(self._xc.copy(),self._zc.copy(), u.copy(), q.copy(), h.copy())))
                self._tsteps.append(n*dt)
                
                if fileName != None:
                    np.save(fileName, verts)
                
                courant = weno.get_Max_Phase_Speed(qbedload, self._zc, self._nP)*dt/self._dx
                surf = self._zc + h
                print('Time step: {0} mins - uavg: {1} - Elevation {2}'.format(n*dt/60., u.mean(), surf.mean()))
                print('Courant number: {0}'.format(courant))
            
            
        return self._zc, h, q, u, surf
            
            
            
            
            
# -------------------------------------------------------------------
'''
class UpwindMorphologicalModel(NullMorphologicalModel):


    def run_model(self, simulationTime, z=None, dt=1, useSlopeAdjust=True):

        print(' Starting simulation....')
        # --------------------------------
        #  Setup the model run parameters
        # --------------------------------
        nt = int(simulationTime / dt)  # Number of time steps

        print('Number of time steps: {0}'.format(nt))

        # --------------------------------
        # Set up the domain, BCs and ICs
        # --------------------------------
        print('Grid dx = {0}'.format(self._dx))
        print('Grid nx = {0}'.format(self._nx))

        zc = None
        if z == None:
            zc = self._z_init.copy()
        else:
            zc = z.copy()

        # Is this the correct way to calculate the bed slope?
        bed_slope = np.gradient(zc, self._dx)
        bedShear = self._bed_shear.copy()

        # --------------------------------
        # Initialize the sed transport
        # --------------------------------
        qbedload = np.zeros(self._nx)

        print('D50:    {0}'.format(self._D50))
        print('Rho Particle:    {0}'.format(self._rho_particule))
        print('Angle Repose Degrees:    {0}'.format(self._angleReposeDegrees))
        print('Max Shear Stress:    {0}'.format(bedShear.max()))

        for i in range(0, self._nx):
            qbedload[i] = sedtrans.get_unit_bed_load_slope_shear(bedShear[i],
                                                                 self._D50,
                                                                 bed_slope[i],
                                                                 self._rho_particule,
                                                                 self._angleReposeDegrees,
                                                                 type=self._type,
                                                                 useSlopeAdjust=useSlopeAdjust)
        #qbedload = savgol_filter(qbedload, 25, 3)

        print('qbedload shape: {0}'.format(qbedload.shape))
        print('Max qbedload = {0}'.format(qbedload.max()))
        # --------------------------------
        # Eq 57
        # --------------------------------
        roe_speed = np.zeros(self._nx)
        flux = np.zeros(self._nx)

        # --------------------------------
        # Set up the model reporting parameters
        # --------------------------------


        # --------------------------------
        #  Run the model
        # --------------------------------
        limiter = np.zeros(self._nx)
        for n in range(1, nt):
            zn = zc.copy()
            for i in range(0, self._nx):  # i=2
                zlocal = weno.get_stencil(zn, i - 2, i + 4)

                r = weno.get_r(zn, i)

                limiter[i] = weno.van_leer(r)

                # Since k=3
                # stencil is i-2 to i+2
                qlocal = weno.get_stencil(qbedload, i - 2, i + 4)

                if len(qlocal) != 6:
                    raise ValueError('Stencil is incorrect')

                # Determine the Upwind flux
                # The 0.5 comes from the c+abs(c) which is 2 if the wave speed is +ive
                # this is the evaluation of the left and right based fluxes. Eq. 18 and 19
                # Note this actually comes from the appendix in Long et al:
                # "If ai + 1/2≥0, the flow is from left to right, and corresponding
                # bedform phase speed is also from left to right. Otherwise, if
                # ai + 1/2b0, the flow is from right to left. Since only the sign of
                # ai+1/2 is needed, we can simply use sign((qi+1 − qi)(zbi+1 − zbi))
                # to avoid division by a small number when (zbi +1−zbi) ap- proaches
                # zero. Because WENO only requires the sign of the phase speed, it is
                # much more stable than schemes that require accurate estimate of phase
                # speed both in magnitude and sign."

                roe_speed[i] = np.sign((qlocal[3] - qlocal[2]) * (zlocal[3] - zlocal[2]))

                if roe_speed[i] >= 0.0:
                    # flux[i] = weno.get_left_flux(qlocal)
                    qlocal = weno.get_stencil(qbedload,i-1,i+1)
                    flux[i] = qlocal[0] + 0.5*limiter[i] *(qlocal[1]-qlocal[0])
                    # flux[i] = qlocal[0]
                else:
                    # flux[i] = weno.get_right_flux(qlocal)
                    qlocal = weno.get_stencil(qbedload, i-1, i+2)
                    # flux[i] = qlocal[1]
                    flux[i] = qlocal[2] + 0.5 * limiter[i] * (qlocal[1] - qlocal[2])
                    #flux[i] = qlocal[2]


            # ------------------------------
            # Need the sign of the phase speed
            # Need to check this out
            # ------------------------------
            for i in range(0, self._nx):  # i=2
                floc = weno.get_stencil(flux, i - 1, i + 1)
                zc[i] = zn[i] - (1. / (1. - self._nP)) * (dt / self._dx) * (floc[1] - floc[0])

            bed_max_delta = np.max(np.abs(zn - zc))

            # ------------------------------
            # Apply the avalanche model
            # ------------------------------
            if self._avalanche:
                zc, iterations1 = avalanche_model(self._dx, self._xc, zc, adjustment_angle=self._adjustment_angle)
                # Now flip it to run in reverse
                zflip = np.flip(zc, axis=0)
                zflip, iterations1 = avalanche_model(self._dx, self._xc, zflip, adjustment_angle=self._adjustment_angle)
                zc = np.flip(zflip, axis=0)

            # ----------------------------------
            # Apply the two-step smooting scheme Eq. 6 in Niemann et al 2011.
            # ----------------------------------
            if self._smooth:
                zhat = np.zeros(self._nx)
                for i in range(0, self._nx):  # i=2
                    zlocal = weno.get_stencil(zc, i - 1, i + 2)
                    zhat[i] = 0.5*zlocal[1] + 0.25*(zlocal[0]+zlocal[2])

                for i in range(0, self._nx):
                    zhatlocal = weno.get_stencil(zhat, i - 1, i + 2)
                    zc[i] = (3./2.)*zhatlocal[1] - 0.25*(zhatlocal[0]+zhatlocal[2])
                # Update the gradient.
                bed_slope = np.gradient(zc, self._dx)
                bed_slope = savgol_filter(bed_slope, 11, 2)
            else:
                bed_slope = np.gradient(zc, self._dx)


            useShearShifter = True
            if useShearShifter == True:
                shift = self.calculate_mean_bedform_shift(self._z_init, zc)
                #shift = 0.5 * (np.mean(shift) + np.max(shift))
                shift = np.max(shift)
                shift = shift/self._dx
                bedShear = sciim.interpolation.shift(self._bed_shear, shift, mode='wrap', order = 1)




            # ------------------------------
            # Update the bed load
            # ------------------------------
            for i in range(0, self._nx):
                qbedload[i] = sedtrans.get_unit_bed_load_slope_shear(bedShear[i],
                                                                 self._D50,
                                                                 bed_slope[i],
                                                                 self._rho_particule,
                                                                 self._angleReposeDegrees,
                                                                 type=self._type,
                                                                 useSlopeAdjust=useSlopeAdjust)
            #qbedload = savgol_filter(qbedload, 25, 3)
        print(' Done')
        print(' ----------------------------')
        return zc, qbedload, bedShear, roe_speed


    def get_point_on_line(self, a, b, c, x0, y0):
        denominator = (a**2 + b**2)
        x1 = (b*(b*x0 - a*y0) - a*c)/denominator
        y1 = (a*(-b*x0 + a*y0) - b*c)/denominator
        d = abs(a*x0 + b*y0 + c)/math.sqrt(denominator)
        return x1, y1, d




    def calculate_mean_bedform_shift(self, zn, znp):
        distance_tol = 1.e-6
        slope_tol = 10.
        StencilWidthBack = 4
        StencilWidthForwards = 4

        translations = []
        for i in range(0, self._nx):

            znlocal = weno.get_stencil(zn, i - StencilWidthBack, i + StencilWidthForwards)
            znplocal = weno.get_stencil(znp, i - StencilWidthBack, i + StencilWidthForwards)
            xlocal = weno.get_stencil(self._xc, i - StencilWidthBack, i + StencilWidthForwards)
            xlocal = np.linspace(0., len(xlocal)*self._dx,len(xlocal))

            # looking for the front of the bed
            # Check if monotinitcally decreasing
            if (np.all(np.diff(znlocal) <= 0) and np.all(np.diff(znplocal) <= 0)):

                resnp = scipy.stats.linregress(xlocal, znplocal)
                a = resnp.slope

                # Calculate the angle of the slope
                slope_angle_degrees = math.atan(abs(a)) * 57.2958
                if slope_angle_degrees > slope_tol:

                    c = resnp.intercept
                    b = -1.
                    x0 = xlocal[StencilWidthBack]
                    y0 = znlocal[StencilWidthBack]

                    x1, y1, d = self.get_point_on_line(a, b, c, x0, y0)

                    if d > distance_tol:
                        #print(x1, y1, d)
                        deltaY = abs(y0 - y1)
                        gamma = math.asin(deltaY/d)
                        movement = d/math.cos(gamma)

                        translations.append(movement)

        translations = np.array(translations)
        return translations


    def update_bed_shear(self, bedShear,z0, zc):
        for i in range(0, self._nx):
            # Calculate the change in the bed
            # A negative value means bed in going up, positive means erosion


            PolyOrder = 2
            StencilWidthBack = 5
            StencilWidthForwards = 2
            zlocal = weno.get_stencil(z0, i - StencilWidthBack, i + StencilWidthForwards)
            bedShearLocal = weno.get_stencil(bedShear, i - StencilWidthBack, i + StencilWidthForwards)
            xlocal = weno.get_stencil(self._xc, i - StencilWidthBack, i + StencilWidthForwards)
            xRealative = np.linspace(0., len(xlocal)*self._dx,len(xlocal))


            try:
                zpoly = np.polyfit(zlocal, xRealative, PolyOrder)
                p = np.poly1d(zpoly)
                xnew = p(zc[i])

                if xnew > xRealative.max() or xnew < xRealative.min():
                    print('Exceeded range.')
                else:
                    tauPoly = np.polyfit(xRealative,bedShearLocal, PolyOrder)
                    p = np.poly1d(tauPoly)
                    bedShear[i] = p(xnew)
            except:
                print('zlocal = {0}'.format(zlocal))
                print('xRealtive = {0}'.format(xRealative))

        return bedShear




def getLimiter(upSlope,dsSlope):
    r = dsSlope/(upSlope + 1.e-12)
    if r < 0.:
        r=0.
    phi = (r + abs(r)) / (1 + abs(r))

    if phi > 2.:
        phi = 2

    if phi < 0.:
        phi = 0

    return phi,r

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



def load_bed_profile(bed_profile_path):
    z = []
    params = {}

    with open(bed_profile_path) as f:
        for line in f:
            values = line.split()
            if len(values) > 0:
                if is_number(values[0]) == False:
                    print(values)
                    if float(values[1]).is_integer():
                        params[values[0]] = int(values[1])
                    else:
                        params[values[0]] = float(values[1])
                else:

                    if is_number(values[0]) == True:
                        z.append(float(values[0]))
    z = np.array(z)
    xmax = float(params['nrows']) * params['cellsize']
    nx = params['nrows']
    dx = params['cellsize']
    xc = np.linspace(0, xmax, nx)

    return z, xc, dx

def load_bed_shear_stress(bed_shear_stress_path):
    retval = []
    with open(bed_shear_stress_path) as f:
        for line in f:
            values = line.split()
            if len(values) > 0:
                retval.append(float(line))

    return np.array(retval)'''


