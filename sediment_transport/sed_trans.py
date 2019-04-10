#!/usr/bin/env python

import numpy as np
import math

G=9.81
rho = 1000.
nu = 10**-6.
kappa = 0.4
phi = 0.7854

def get_bed_porosity(D0):
    ''' EQ 12 from Wu, W., Wang, S.S.Y (2007) One-Dimensional Modelling of Dam Break Flow
        over Movable Beds'''
    pm = 0.13 + 0.21/(D0 + 0.002)**0.21
    return pm

def get_gammS(rho_particule):
    return G*(rho_particule-rho)

def get_Xi(D0,rho_particule):
    '''
    Equation 1.31
    '''
    gammaS = get_gammS(rho_particule)
    xi = ((gammaS * D0**3.)/(rho * nu**2.))**(1./3.)
    return xi

def get_Ycr(D0,rho_particule):
    '''
    Equation 1.34
    '''
    xi = get_Xi(D0,rho_particule)
    ycr = 0.13*(xi**(-0.392))*math.exp(-0.015*xi**2.) + 0.045*(1-math.exp(-0.068*xi))
    return ycr

def get_critical_shearvelocity(D0,rho_particule):
    '''
    Equation 1.30
    '''
    gammaS = get_gammS(rho_particule)
    ycr = get_Ycr(D0,rho_particule)
    u_cr = math.sqrt(ycr*gammaS*D0/rho)
    return u_cr

def get_critical_shear(D0, rho_particule):
    u_cr = get_critical_shearvelocity(D0, rho_particule)
    tau_cr = rho*u_cr**2.0
    return tau_cr

def get_cf(h,D0):
    '''
    @param depths: array of depths
    @param D0: median grain size in m
    @return: an array of the the coefficient of friction
    '''
    ks = 2.*D0
    c_f = (1./kappa)*math.log1p(11.*h/ks)
    return c_f

def get_bed_shear(h,u,D0):
    cf = get_cf(h,D0)
    tau_bed = rho*(u/cf)**2.
    return tau_bed

def get_Y(h,u,D0, rho_particule):
    gammaS = get_gammS(rho_particule)
    tau = get_bed_shear(h,u,D0)
    y = tau/(gammaS*D0)
    return y

def get_Y(shear, D0, rho_particule):
    gammaS = get_gammS(rho_particule)
    y = shear / (gammaS * D0)
    return y

#def get_unit_bed_load(h,u,D0,rho_particule,method='mpm'):
def get_unit_bed_load(**kwargs): 
    
    h = kwargs.get('h',None)
    u = kwargs.get('u',None)
    tau = kwargs.get('tau',2)
    
    D0 = kwargs.get('D0',0.001)
    rho_particule = kwargs.get('rho_particule',2650)
    method = kwargs.get('method','mpm')
    
    
    y = 0
    sign = 1
    gammaS = get_gammS(rho_particule)
    if tau == None:
        ''' Using Meyer-Peter and Muller '''
        
        if u < 0:
            sign = -1
        u = abs(u)
        y = get_Y(h,u,D0, rho_particule)
    
    else:
        y = abs(tau)/(gammaS*D0)
        if tau < 0:
            sign = -1
    
    y_cr = get_Ycr(D0, rho_particule)
    

    qsb = 0
    phi = 0
    if y > y_cr:
        if method=='bagnold':
            phi = (0.5*8.5)*math.sqrt(y)*(y-y_cr)
        elif method == 'mpm':
            if y > 0.047:
                phi = 8.*(y-0.047)**(1.5)
        qsb = (math.sqrt(gammaS)*(D0**1.5)/math.sqrt(rho)) * phi

    return qsb*sign


def get_unit_bed_load2(h,u,D0,rho_particule,type='mpm'):
    ''' Using Meyer-Peter and Muller '''
    sign = 1
    if u < 0:
        sign = -1
    u = abs(u)
    tau = get_bed_shear(h,u,D0)
    y = get_Y(tau,D0, rho_particule)
    y_cr = get_Ycr(D0, rho_particule)
    gammaS = get_gammS(rho_particule)

    qsb = 0
    phi = 0
    if y > y_cr:
        if type=='bagnold':
            phi = (0.5*8.5)*math.sqrt(y)*(y-y_cr)
        elif type == 'mpm':            
            phi = 8.*(y-y_cr)**(1.5)
        qsb = (math.sqrt(gammaS)*(D0**1.5)/math.sqrt(rho)) * phi

    return qsb*sign


def get_unit_bed_load_slope_shear(shear, D0, slope, rho_particule, angleReposeDegrees=30.0, type='bagnold', useSlopeAdjust=True):
    ''' Using Meyer-Peter and Muller or Bagnold but adjusted for the bed slope based on eq. 2.10 from my proposal'''
    sign = 1
    if shear < 0:
        sign = -1

    shear = abs(shear)

    y = get_Y(shear, D0, rho_particule)
    y_cr = get_Ycr(D0, rho_particule)
    gammaS = get_gammS(rho_particule)

    qsb = 0.
    phi = 0.

    slopeAdjustment = 0.0
    if useSlopeAdjust == True:
        slopeAdjustment = (y_cr / math.tan(math.pi * angleReposeDegrees / 180.0)) * slope

    y_cr_modified = (y_cr + slopeAdjustment)

    if y > y_cr_modified:
        if type == 'bagnold':
            phi = (0.5 * 8.5) * math.sqrt(y) * (y - y_cr_modified)
        elif type == 'mpm':
            phi = 8.0 * (y - y_cr_modified) ** 1.5
        else:
            raise ValueError('Wrong value for sediment transport provided.')
        qsb = (math.sqrt(gammaS) * (D0 ** 1.5) / math.sqrt(rho)) * phi
        

    return qsb * sign


def get_unit_bed_load_slope(h,u,D0, slope, rho_particule, angleReposeDegrees = 30.0, type='mpm', useSlopeAdjust=True):
    ''' Using Meyer-Peter and Muller but adjusted for the bed slope based on eq. 2.10 from my proposal'''
    sign = 1
    if u < 0:
        sign = -1
    u = abs(u)

    tau = get_bed_shear(h,u,D0)
    
    return get_unit_bed_load_slope_shear(tau, D0, slope, rho_particule, angleReposeDegrees, type, useSlopeAdjust)
    
    
    
    
    
    
    
    
    
    y = get_Y(tau,D0, rho_particule)
    y_cr = get_Ycr(D0, rho_particule)
    gammaS = get_gammS(rho_particule)

    qsb = 0.
    phi = 0.
    slopeAdjustment = (y_cr / math.tan(math.pi * angleReposeDegrees / 180.0))*slope
    
    y_cr_modified = (y_cr+slopeAdjustment)
    
    if useSlopeAdjust==False:
        y_cr_modified = y_cr
    
    if y > y_cr_modified:
        if type=='bagnold':        
            phi = (0.5*8.5)*math.sqrt(y)*(y-y_cr_modified)
        elif type == 'mpm':
            phi = 8.0 *  (y-y_cr_modified)**1.5
        qsb = (math.sqrt(gammaS)*(D0**1.5)/math.sqrt(rho)) * phi

    return qsb*sign


def get_upwind_bedload_flux_neq(Qbi, Qbip1,Qi):
    Qb=0
    if Qi >=0:
        Qb = Qbi
    else:
        Qb = Qbip1
    return Qb


def get_van_rijn_t(U, h, D50, D90, rho_particule):
    ''' Equation 3.57 in Wu, 2007'''
    Ustar = U*math.sqrt(G)/(18.0*math.log(4.0*h/D90))
    Ucrit = get_critical_shearvelocity(D50,rho_particule)
    T = ((Ustar/Ucrit)**2) - 1
    return T


def get_Ubed(U, h, D50, rho_particule,D90=0):
    '''
    @return: Returns the bedload velocity based on equation 3.136 in Wu 2007
    '''
    ub = 0.0
    if D90 == 0.0:
        D90 = D50
    T = get_van_rijn_t(U, h, D50, D90, rho_particule,)
    print( 'T %s' % T )
    if T >= 0.0:
        ub = math.sqrt( (rho_particule/rho -1) * G * D50 ) * 1.5 * (T)**0.6
    return ub


def get_upwind_bedload_flux(hi,hip1,ui,uip1,D0, rho_particule, type='mpm'):
    qsb=0.
    sign = 1.
    '''
    This is the flux at i + 1/2
    EQ 5
    '''
    if ui >= 0:
        qsb = get_unit_bed_load(hi,abs(ui),D0,rho_particule,type)
    else:
        qsb = get_unit_bed_load(hip1,abs(uip1),D0,rho_particule,type)
        if uip1 < 0:
            sign = -1.
    return qsb * sign

def get_quick_bedload_flux(hWW, hW, hP, hE, uWW, uW, uP, uE, D50, rho_sediment, sed_model):
    qsb=0.
    sign = 1.
    '''
    This is the flux at i + 1/2
    EQ 4.119 in Wu 2007
    '''
    Fw = 0.5 * (uW + uP)

    if Fw >= 0:
        qsbWW = get_unit_bed_load(hWW,abs(uWW),D50,rho_sediment,sed_model)*cmp(uWW,0)
        qsbW = get_unit_bed_load(hW,abs(uW),D50,rho_sediment,sed_model)*cmp(uW,0)
        qsbP = get_unit_bed_load(hP,abs(uP),D50,rho_sediment,sed_model)*cmp(uP,0)
        qsb = (-1.0/8.0)*qsbWW + 0.75 * qsbW + (3.0/8.0) * qsbP

    else:
        qsbW = get_unit_bed_load(hW,abs(uW),D50,rho_sediment,sed_model)*cmp(uW,0)
        qsbP = get_unit_bed_load(hP,abs(uP),D50,rho_sediment,sed_model)*cmp(uP,0)
        qsbE = get_unit_bed_load(hE,abs(uE),D50,rho_sediment,sed_model)*cmp(uE,0)
        qsb = (3.0/8.0)*qsbW + 0.75 * qsbP - (1.0/8.0) * qsbE

    return qsb

def get_ubar(S,h,D0):
    cf = get_cf(h,D0)
    ubar = cf*math.sqrt(G*S*h)
    return ubar

if __name__ == "__main__":
    D0 = 0.001
    h = 2.0
    S = 0.002
    u = 4.609453777

    Ycr = 0.037
    Xi = 25.296
    qsb = 0.001220
    Y = 2.424
    rho_particule = 2650

    ''' Test against following'''
    S = 1./1500.0
    h = 2.0
    D0 = 0.002
    u = get_ubar(S,h,D0)
    Y = get_Y(h,u,D0,rho_particule)
    print('Expected Y: 0.4043, got %s' % Y)
    qsb = get_unit_bed_load(h,u,D0, rho_particule, type='mpm')
    print('Expected bedload: 6.15 x 10-4, got: %s' % qsb)


    D0 = 0.001
    h = 2.0
    S = 0.002
    u = 4.609453777

    Ycr = 0.037
    Xi = 25.296
    qsb = 0.001220
    Y = 2.424
    rho_particule = 2650


    calc_y = get_Y(h,u,D0,rho_particule)

    ustar_cr = 0.024
    cal_xi = get_Xi(D0,rho_particule)
    cal_ycr = get_Ycr(D0,rho_particule)
    cal_ustar_cr = get_critical_shearvelocity(D0,rho_particule)
    cal_qsb = get_unit_bed_load(h,u,D0,rho_particule,'bagnold')

    print( 'Expected Xi = %s, Returned Xi = %s ' % (Xi,cal_xi ))
    print( 'Expected Ycr = %s, Returned Ycr = %s ' % (Ycr,cal_ycr ))
    print( 'Expected u*cr = %s, Returned u*cr = %s ' % (ustar_cr,cal_ustar_cr ))
    print( 'Expected y = %s,  Returned y= %s' %(Y,calc_y))
    print( 'Expected qsb = %s,  Returned qsb= %s' %(qsb,cal_qsb))




