''' This is a library of differnet geometries that can be used for the modelling '''
import csv
import copy
import numpy as np
import math
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
    
if __name__ == "__main__":
    x,zb = single_hump(20., 201)
    print zb
