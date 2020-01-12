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


'''
This is a modudel containing different implementations of the Exner Models for Parameterized Models

'''


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
        
        qbedload1 = baseModel._calculate_bedload(baseModel._h, baseModel._u, baseModel._xc, z1, baseModel._time)
        #qbedload1 = baseModel._calculate_bedload(z1)
        
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
