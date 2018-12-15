#! /usr/bin/env python
# -*- coding: utf-8 -*

import cython
import os
import schemes.weno as weno
import sediment_transport.sed_trans as sedtrans
from schemes.avalanche_scheme import avalanche_model, get_slope
import numpy as np
from scipy.signal import savgol_filter


class NullMorphologicalModel(object):
    """

    """

    def __init__(self, D50, rho_particule = 2650., angleReposeDegrees = 30., nP=0.4):
        self._D50 = D50
        self._rho_particule = rho_particule
        self._angleReposeDegrees = angleReposeDegrees
        self._nP=nP
        self._type = 'bagnold'


    '''
    '''
    def setup_model(self, bed_shear, z_init, xc, dx, useAvalanche = True, useSmoother = True, adjustment_angle=29.):
        self._z_init = z_init
        self._bed_shear = bed_shear
        self._xc = xc
        self._nx = len(xc)
        self._dx = dx
        self._avalanche = useAvalanche
        self._smooth = useSmoother
        self._adjustment_angle = adjustment_angle

    def set_sed_trans_model(self, model):
        self._type = model

    def run_model(self, simulationTime, dt = 1):
        pass



class WenoMorphologicalModel(NullMorphologicalModel):



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
        for n in range(1, nt):
            zn = zc.copy()
            for i in range(0, self._nx):  # i=2
                zlocal = weno.get_stencil(zn, i - 2, i + 4)
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
                    flux[i] = weno.get_left_flux(qlocal)
                else:
                    flux[i] = weno.get_right_flux(qlocal)

            # ------------------------------
            # Need the sign of the phase speed
            # Need to check this out
            # ------------------------------
            for i in range(0, self._nx):  # i=2
                floc = weno.get_stencil(flux, i - 1, i + 3)
                zlocal = weno.get_stencil(zn, i - 1, i + 4)


                zc[i] = zn[i] - (1. / (1. - self._nP)) * dt / self._dx * (floc[1] - floc[0])

            bed_max_delta = np.max(np.abs(zn - zc))

            # ------------------------------
            # Apply the avalanche model
            # ------------------------------
            zc, iterations1 = avalanche_model(self._dx, self._xc, zc, adjustment_angle=self._adjustment_angle)
            # Now flip it to run in reverse
            zflip = np.flip(zc, axis=0)
            zflip, iterations1 = avalanche_model(self._dx, self._xc, zflip)
            zc = np.flip(zflip, axis=0)

            # ----------------------------------
            # Apply the two-step smooting scheme Eq. 6 in Niemann et al 2011.
            # ----------------------------------
            zhat = np.zeros(self._nx)
            for i in range(0, self._nx):  # i=2
                zlocal = weno.get_stencil(zc, i - 1, i + 2)
                zhat[i] = 0.5*zlocal[1] + 0.25*(zlocal[0]+zlocal[2])

            for i in range(0, self._nx):
                zhatlocal = weno.get_stencil(zhat, i - 1, i + 2)
                zc[i] = (3./2.)*zhatlocal[1] - 0.25*(zhatlocal[0]+zhatlocal[2])


            # Update the gradient.
            bed_slope = np.gradient(zc, self._dx)

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
        return zc, qbedload, bed_slope


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
                zlocal = weno.get_stencil(zn, i - 1, i + 4)
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

            '''
            Slope limiter approach for modifying the bed shear stress profile.
            '''
            useSlopeLimiter = False
            if useSlopeLimiter == True:
                for i in range(0, self._nx):
                    zlocal = weno.get_stencil(zc, i - 1, i + 2)
                    bedShearLocal = weno.get_stencil(bedShear, i - 1, i + 2)

                    if bedShear[i] >= 0.0:
                        upSlope = zlocal[1] - zlocal[0]
                        dsSlope = zlocal[2] - zlocal[1]

                        #bedShearLocal = weno.get_stencil(self._bed_shear, i - 1, i + 2)

                        slopelimiter,r = getLimiter(upSlope, dsSlope)
                        bedShear[i]= bedShearLocal[1] + 0.5* slopelimiter *(bedShearLocal[0] - bedShearLocal[1])
                    else:
                        upSlope = zlocal[2] - zlocal[1]
                        dsSlope = zlocal[1] - zlocal[0]
                        slopelimiter, r = getLimiter(upSlope, dsSlope)
                        bedShear[i] = bedShearLocal[1] + 0.5 * slopelimiter * (bedShearLocal[2] - bedShearLocal[1])


                    # print(i,r,slopelimiter, bedShear[i],self._bed_shear[i] )



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


'''
Loads up the bed profile
'''
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

    return np.array(retval)