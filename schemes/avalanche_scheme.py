#!/usr/bin/env python
'''
Avalanche scheme based on the method described in Niemann et al. 2011.

Does not work with negative bed elevations.
'''
import numpy as np
import math
from scipy.optimize import brentq,bisect,newton
from schemes.weno import get_stencil

def get_slope(dx,zloc):
    '''
    Returns the slopes in the downstream direction.
    dx: delta x
    zloc: a list containing two values
    '''
    if len(zloc)!=2:
        raise ValueError('The variable zloc must contain two values')
    
    dz = zloc[1]-zloc[0]
    return math.atan(dz/dx)*180./math.pi




def avalanche_model(dx,xc,z,max_iterations = 100,threshold_angle=30.5,angle_of_repose=30.0,adjustment_angle=29.5):
    '''
    
    '''
    z_current = z.copy()
    iteration = 0
    slopes = np.zeros(len(z_current))
    for i in range(len(xc)):
            zloc = get_stencil(z_current,i,i+2)
            slopes[i] =get_slope(dx, zloc)
    
    if min(slopes) < threshold_angle*-1.0:
        z_new = z.copy()
        
        
        bed_changed = True
        
    
        while bed_changed == True and iteration < max_iterations:
            bed_changed = False
            iteration +=1
            for i in range(len(xc)):
                zloc = get_stencil(z_current,i,i+2)
                slope =get_slope(dx, zloc)
                
                factor = 0.1
                check_adjustment = True
                if slope < angle_of_repose*-1.0:
                    #print('Checking slope for i={0} slope={1}'.format(i,slope))
                    xloc = get_stencil(xc,i-1,i+2)
                    zloc = get_stencil(z_current,i-1,i+2)        
                    area_old = get_area_polygon(dx, zloc)
        
                    dz = adjustment_to_target(dx,zloc,adjustment_angle)
                    
                    if iteration < 75:
                        z_new[i]-=dz * factor
                    else:
                        z_new[i]-=dz * 0.9
        
                    zloc_new = get_stencil(z_new,i-1,i+2)          
                    area_new = get_area_polygon(dx, zloc_new)
                    del_area = area_old - area_new
        
                    #print('Old area: {0}, New area: {1}'.format(area_old,area_new))
                    #print('Difference in area: {0}'.format(del_area))
        
                    # move downstream
                    xloc_p1 = get_stencil(xc,i,i+3)
                    zloc_p1 = get_stencil(z_current,i,i+3)
                    
                    #print('i={0}, del_area:{1}, zloc={2}'.format(i,del_area,zloc_p1))
                    
                    if del_area > 0:
                        new_z = adjust_bed(dx,xloc_p1,zloc_p1,del_area)
                        if i+1 == len(z_current):
                            z_new[0] = new_z
                        else:
                            z_new[i+1] = new_z
                    bed_changed = True
        
                z_current = z_new.copy()   

    return z_current,iteration

def adjustment_to_target(dx,zloc, angle_of_repose = 29.5):
    dz_target = dx * math.tan(angle_of_repose*math.pi/180.)
    
    dz_current = zloc[1]-zloc[2]
    
    return dz_current - dz_target


def adjust_bed(dx,xloc,zloc,del_area):
    '''
    Adjusts the elevation of the bed due to the avalanche.
    '''
    if len(xloc)!=3:
        raise ValueError('The variable xloc must contain three values')
    
    if len(zloc)!=3:
        raise ValueError('The variable zloc must contain three values')
    
     
    current_area = get_area_polygon(dx,zloc)
    
    target_area = current_area + del_area
    
    z_new = bisect(f,-1.0,100.,args=(dx,zloc,target_area))
    
    return z_new
    
    
    

def f(z_target,dx,zloc,target_area):
    retval = 0.0
    
    X = np.zeros(7)
    X[0] = 0.0
    X[1] = 1.0*dx
    X[2] = 2.0*dx
    X[3] = dx*2.
    X[4] = dx
    X[5] = 0.0
    X[6] = X[0]
        
    Y = np.zeros(7)
    Y[0:3]=zloc
    Y[1] = z_target
    Y[3:6]=[0.0,0.0,0.0]
    Y[6]=Y[0]
    
    numPoints = len(Y)
    j = numPoints-1
    
    for i in range(numPoints):
        retval+=  (X[j]+X[i]) * (Y[j]-Y[i])
        j=i
        
    area = retval/2.0
    return area - target_area
    
    
        
        
    

def get_area_polygon(dx,zloc):
    ''' 
    
    http://www.mathopenref.com/coordpolygonarea2.html 
    '''
    
    retval = 0.0
    
    X = np.zeros(7)
    X[0] = 0.0
    X[1] = 1.0*dx
    X[2] = 2.0*dx
    X[3] = dx*2.
    X[4] = dx
    X[5] = 0.0
    X[6] = X[0]
        
    Y = np.zeros(7)
    Y[0:3]=zloc
    Y[3:6]=[0.0,0.0,0.0]
    Y[6]=Y[0]
    
    numPoints = len(Y)
    j = numPoints-1
    
    for i in range(numPoints):
        retval+=  (X[j]+X[i]) * (Y[j]-Y[i])
        j=i
        
    return retval/2.0



def get_area_bed(dx,X,Z):
    numPoints = len(Z)
    j = numPoints-1
    retval = 0.0
    for i in range(numPoints):
        retval+=  (X[j]+X[i]) * (Z[j]-Z[i])
        j=i
        
    return retval/2.0
        
def get_area(xloc,zloc):
    '''
    http://www.mathopenref.com/coordtrianglearea.html
    '''
    if len(xloc)!=3:
        raise ValueError('The variable xloc must contain three values')
    
    if len(zloc)!=3:
        raise ValueError('The variable zloc must contain three values')
    
    Ax = xloc[0]
    Bx = xloc[1]
    Cx = xloc[2]
    
    Ay = zloc[0]
    By = zloc[1]
    Cy = zloc[2]
    
    area = abs( (Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))/2.0 )
    
    return area