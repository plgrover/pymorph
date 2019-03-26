''' Written as a function '''
import copy
import numpy as np
import math
from scipy.integrate import simps


def morphmodel_2ndO_upwind(x,zb,depth,qf,timesteps,dt,use_slope_correction=True):

    dx = x[1] - x[0]

    number_nodes = len(x)
    z = np.ones(number_nodes) * depth
    qbs = np.zeros(number_nodes)
    grdqb = np.zeros((number_nodes,2))

    #qf = 0.45
    A = 1.0
    u = qf/(z-zb)

    print( 'here 2')

    # This is a simple sediment transport model
    sf = calculate_slope_factor(x,zb)
    qbs = (A * u) * sf

    print('Number of timesteps: %s' % timesteps)
    for t in range(timesteps):
        zbn = copy.deepcopy(zb)
        for stage in range(2):
            ''' ----------------------------- '''
            ''' Calculate the derivative      '''
            ''' ----------------------------- '''
            for i in range(2,number_nodes-1):
                ''' Using a  second order upwind difference equation '''
                grdqb[i,stage] = (1./(6. * dx))*(2.*qbs[i+1] + 3.*qbs[i] - 6.*qbs[i-1] + qbs[i-2])

                if abs(grdqb[i,stage]) > 0.01:
                    grdqb[i,stage] = (1./(dx))*(qbs[i]-qbs[i-1])

            ''' Update the boundary and near-boundary conditions '''
            ''' i = 1'''
            grdqb[1,stage] = (1./(6. * dx))*(2.*qbs[2] + 3.*qbs[1] - 6.*qbs[0] + qbs[number_nodes-1])
            ''' i = 0'''
            grdqb[0,stage] = (1./(6. * dx))*(2.*qbs[1] + 3.*qbs[0] - 6.*qbs[number_nodes-1] + qbs[number_nodes-2])
            ''' i = number_nodes -1 '''
            grdqb[number_nodes-1,stage] = (1./(6. * dx))*(2.*qbs[0] + 3.*qbs[number_nodes-1] - 6.*qbs[number_nodes-2] + qbs[number_nodes-3])

            for i in range(number_nodes):
                if stage == 1:
                    zb[i] = zbn[i] - dt*grdqb[i,0]
                else:
                    zb[i] = zbn[i] - 0.5*dt*(grdqb[i,0] +  grdqb[i,1])

            ''' Update the velocity and bedload '''
            h = z - zb
            u = qf/h
            qbs = (A * u)
            if use_slope_correction == True:
                sf = calculate_slope_factor_dey(x,zb)
                qbs = qbs * sf


    return x,zb,u,qbs

def morphmodel_upwind(x,zb,depth,qf,timesteps,dt,use_slope_correction=True):

    dx = x[1] - x[0]

    number_nodes = len(x)
    z = np.ones(number_nodes) * depth
    qbs = np.zeros(number_nodes)
    grdqb = np.zeros((number_nodes,2))

    #qf = 0.45
    A = 1.0
    u = qf/(z-zb)

    # This is a simple sediment transport model
    sf = calculate_slope_factor(x,zb)
    qbs = (A * u) * sf

    print('Number of timesteps: %s' % timesteps)
    for t in range(timesteps):
        zbn = copy.deepcopy(zb)
        for stage in range(2):
            ''' ----------------------------- '''
            ''' Calculate the derivative      '''
            ''' ----------------------------- '''
            for i in range(1,number_nodes):
                grdqb[i,stage] = (1./(dx))*(qbs[i]-qbs[i-1])


            ''' Update the boundary and near-boundary conditions '''
            ''' i = 1'''
            grdqb[1,stage] = (1./(dx))*(qbs[0]-qbs[number_nodes-1])


            for i in range(number_nodes):
                if stage == 1:
                    zb[i] = zbn[i] - dt*grdqb[i,0]
                else:
                    zb[i] = 0.5*(zb[i] + zbn[i])  - 0.5*dt*(grdqb[i,0] +  grdqb[i,1])

            ''' Update the velocity and bedload '''
            h = z - zb
            u = qf/h


            qbs = (A * u)
            if use_slope_correction == True:
                sf = calculate_slope_factor_dey(x,zb)
                qbs = qbs * sf

    return x,zb,u,qbs

def calculate_bed_slope(x,zb):
    slope = np.zeros(len(zb))
    for i in range(len(zb)-1):
        slope[i] =(zb[i+1] - zb[i-1])/(x[i+1] - x[i-1])

    num_node = len(zb)
    slope[num_node-1] =(zb[1] - zb[num_node-2])/(x[1] - x[num_node-2])
    return slope

def calculate_bed_slope_angle(x,zb):
    slope = np.zeros(len(zb))
    for i in range(len(zb)-1):
        slope[i] =max( (math.atan(zb[i] - zb[i-1])/(x[i-1] - x[i])) ,
            math.atan(zb[i+1] - zb[i])/(x[i] - x[i+1]) )
    num_node = len(zb)-1
    slope[num_node] =max( math.atan(zb[num_node] - zb[num_node-1])/(x[num_node-1] - x[num_node]) ,
        math.atan(zb[0] - zb[num_node])/(x[num_node] - x[0]))



    return slope

def calculate_slope_factor_paarl(x,zb,ang_repose=33.0):
    ang_repose_rads = ang_repose * math.pi / 180.
    ang_repose_slope = math.atan(ang_repose_rads)
    ''' This is eq 10 '''
    slope_param = 1./math.tan(ang_repose_rads)

    sf = np.zeros(len(x))
    slope = calculate_bed_slope(x,zb)
    for i in range(len(x)):
        theta = math.atan(slope[i])
        dzdx = slope[i]
        if abs(theta) >= ang_repose_rads:
            dzdx = ang_repose_slope*cmp(theta,0)

        sf[i] = (1. + slope_param*dzdx)**-1.
    return sf


def calculate_slope_factor_dey(x,zb,ang_repose=33.0):
    ''' =COS(C3)*(1-(TAN(C3)/TAN($D$1)))'''
    ang_repose_rads = ang_repose * math.pi / 180.
    slope = calculate_bed_slope(x,zb)
    sf = np.zeros(len(x))
    for i in range(len(x)):
        theta = math.atan(slope[i])
        if abs(theta) >= ang_repose_rads:
            theta = ang_repose_rads*cmp(theta,0)
        sf[i] = math.cos(theta) * (1. - (math.tan(theta)/math.tan(ang_repose_rads) ) )
    return sf





def calculate_slope_factor(x,zb,ang_repose=33.0):
    sf = np.zeros(len(x))
    slope = calculate_bed_slope_angle(x,zb)
    tan_ang_repose = math.tan(ang_repose * math.pi / 180.)
    for i in range(len(x)):
        tan_slope = math.tan(slope[i])
        cos_slope = math.cos(slope[i])
        if tan_slope >= tan_ang_repose:
            sf[i] = 31.3
        else:
            sf[i] = tan_ang_repose / ((tan_ang_repose-tan_slope) * cos_slope)
    return sf

def cmp(a, b):
    return (a > b) - (a < b)

if __name__ == "__main__":
    import morph_geom_lib
    import matplotlib.pyplot as plt

    x,zb = morph_geom_lib.single_hump( 20., 101 )
    zb0 = copy.deepcopy(zb)

    ''' These are the settings from Cowles 2013 '''
    depth = 3.0
    qf = 1.0
    tf = 3.01
    dt = 0.0001
    timesteps = int(tf/dt)
    x,zb,u,qbs = morphmodel_upwind(x,zb,depth,qf,timesteps,dt,True)

    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(211)
    ax1.plot(x,zb, label='bed')
    ax1.plot(x,zb0, label='initial bed')

    slope = calculate_bed_slope(x,zb)
    sf = calculate_slope_factor(x,zb)

    ax2 = fig.add_subplot(212)
    ax2.plot(x,slope,'r')
    ax2.plot(x,sf,'b')
    plt.show()

