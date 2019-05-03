''' This is a library of differnet geometries that can be used for the modelling '''
import os
import csv
import copy
import numpy as np
import pandas as pd
import math
import collections
from scipy.interpolate import interp1d

def single_hump(length, number_nodes):
    L = length
    dx = L/(number_nodes-1)
    
    x = np.zeros(number_nodes)
    zb = np.ones(number_nodes)
    
    ''' Gaussian curve parameters '''
    A0 = 1.0
    A1 = 1.0 # mean
    lam = length
    
    for i in range(len(x)):
        x[i] = -lam/2. + i*dx
        zb[i] = A0 + A1*math.cos(2.0*math.pi*x[i]/lam) 
    
    return x,zb

def flume_experiment(num_cells):   
            
    x=[]
    zb=[]
    with open(r'C:\Users\Patrick\Dropbox\PhD\Morphodynamic_Model\data\BedProfile_eq_v03.csv','rb') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            x.append(float(row[0])/100.)
            zb.append(float(row[1])/100.)
            
    xnew = np.linspace(min(x),max(x), num=num_cells)
    f1 = interp1d(x,zb)
    zbnew = f1(xnew)
    
    ''' Copy the variables '''
    x = copy.deepcopy(xnew)
    zb = copy.deepcopy(zbnew)
    
    return x,zb
    
    
    
   
def readQueensFlume(filepath, resolution=1):
    retval = collections.OrderedDict()
    z = []
    with open(filepath) as f:
        for line in f:
            values = line.split()
            if is_number(values[0])==False:
                if float(values[1]).is_integer():
                    retval[values[0]]=int(values[1])
                else:
                    retval[values[0]]=float(values[1])
            else:
                z.append(float(values[0]))
    z = np.array(z)
    print('Z: {0}'.format(len(z)))
    xmax = float(retval['nrows']) * retval['cellsize']
    nx = retval['nrows']
    dx = retval['cellsize']
    #--------------------------------
    # Increase the resolution on the grid
    #--------------------------------
    x = np.linspace(0, nx*dx, num=len(z))
    f = interp1d(x, z)
    xnew = np.linspace(0, nx*dx, num=len(z)*resolution)
    znew = f(xnew)
    nx = len(xnew)

    return xnew,znew
    
    
# ------------------------

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

'''
Reads a single file from the experimental data from Wiebe
'''
def readWiebeFile(filepath, resolution = 0.5):
    xs = []
    zs = []
    with open(filepath) as f:
        for line in f:
            values = line.split(',')
            if is_number(values[0]) and is_number(values[1]):
                x = float(values[0])/100.
                z = float(values[1])/100.
                xs.append(x)
                zs.append(z)
                
    xs = np.array(xs)
    zs = np.array(zs)
    
    #--------------------------------
    # Increase the resolution on the grid
    #--------------------------------
    xmax = 12.0 #xs.max()
    nx = len(xs)
    f = interp1d(xs, zs)
    
    xnew = np.linspace(0., xmax, num=800)
    znew = f(xnew)
    return xnew, znew

'''
Reads multiple files from the experimental data from Wiebe

Returns a dataframe with the profiles
'''
def readWiebeFiles(profile_folder, resolution = 0.5):
    
    profile_times = [-15, -2, 1, 4, 10, 16, 23, 30, 39, 48, 56, 67, 77, 88, 97, 108, 118, 128, 139, 149, 160, 170, 180, 190, 200, 211, 222, 233, 244, 255, 266, 277, 289, 300, 301]
    
    xprofile = None
    zdict = {}

    for filename in os.listdir(profile_folder):
        name = filename.split('.')[0]
        profile = None
        if len(name)==10:
            profile = '0{0}'.format(name[-1:])
        else:
            profile = name[-2:]

        x, z = readWiebeFile(os.path.join(profile_folder,filename), resolution=0.25)
        
        xprofile = x
        zdict[profile_times[int(profile)]] = z
    
    
    profileDf = pd.DataFrame(zdict)
    profileDf.index = xprofile

    profileDf = profileDf.reindex(sorted(profileDf.columns), axis=1)
    
    return profileDf
    
    
    
    
if __name__ == "__main__":
    x,zb = single_hump(20., 201)
    print(zb)

    
    