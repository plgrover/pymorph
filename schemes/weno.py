'''


The main reference used in this code is as follows:

Long, W., Kirby, J.T., & Shao Z. (2008) A numerical scheme for morphological bed level calculations. 
Coastal Engineering, Vol 55, 167-180. 


http://scicomp.stackexchange.com/questions/20054/implementation-of-1d-advection-in-python-using-weno-and-eno-schemes

'''

import numpy as np




def get_phase_speed(porosity,a,b,u,h):
    '''
    Phase speed of the bedform.
    Long et al. (2008) 
    Eq. 61
    Used for generating an analytical solution for the gaussian hump
    where the transport rate is a function of the current speed.
    q=au**b
    '''
    C = ( (1.0)/( (1.0 - porosity)*(h) ) ) * a * b * u**b
    
    return C
    

def get_exact_solution(zb_0,x_0, porosity,s,a,b,Q,dt):
    '''
    Uses a Lagrangian approach to calculate the elevation of the bed
    based on the approach described in Johnson and Zyserman (2002).
    
    x(z) = x_0(z) + C(z) * dt
    
    The exact solution for this case can be calculated using a
    Lagrangian approach with constant values of z carried
    with celerity Cx(z). The position of a z-point initially
    located at x0(z) after a given time DT, can be found as
    x(z)=x0(z) + C(z) * DT. Thus, the hump is advected in
    the direction of the flow by different celerities at
    different points on the hump
    
    
    Used for generating an analytical solution for the gaussian hump
    where the transport rate is a function of the current speed.
    q=au**b
    
    '''
    zb = []
    x = []
    
    h = s - zb_0
    u = Q/h
    
    z_max = max(zb_0)
    
    for i in range(len(zb_0)):        
        
        C = get_phase_speed(porosity,a,b,u[i],h[i])
        x_new = x_0[i] + C*dt
        #if zb_0[i] < z_max:        
        x.append(x_new)
        zb.append(zb_0[i])
        #else:
        #    break
    
    return np.array(x),np.array(zb)



def get_stencil(ulist,start_index,end_index):
    '''
    Method to get the stencil - implements a periodic BC
    '''
    retval = None
    if end_index > len(ulist):
        a = ulist[start_index:]
        b = ulist[:end_index-len(ulist)]
        retval = np.concatenate((a,b),axis=0)
    elif start_index < 0:
        a = ulist[start_index:]
        #print('First {0}'.format(a)) 
        b = ulist[0:end_index]
        #print('Second: {0}'.format(b))
        retval = np.concatenate((a,b),axis=0)
    else:
        retval = ulist[start_index:end_index]
    
    return retval



def get_left_flux(qloc):
    '''
    Note that qloc should be a stencil with 6 elements even though the last
    element will not be used in the calculation.
    '''
    
    if len(qloc) <> 6:
        raise ValueError('Stencil is not 6')
    
    vareps= 1e-6
    S = np.zeros(3)
    alpha = np.zeros(3)
    omega = np.zeros(3)
    qs = np.zeros(3)
    
    # Calculate smoothness measurements
    S[0] = (13./12.) * (qloc[0] - 2.0*qloc[1] + qloc[2])**2 + (1.0/4.0)* (qloc[0]-4.0*qloc[1] + 3.0*qloc[2])**2 
    
    S[1] = (13./12.) * (qloc[1] - 2.0*qloc[2] + qloc[3])**2 + (1.0/4.0)* (qloc[1] - qloc[3])**2 
    
    S[2] = (13./12.) * (qloc[2] - 2.0*qloc[3] + qloc[4])**2 + (1.0/4.0)* (3.0*qloc[2] - 4.0*qloc[3] + qloc[4])**2

    
    # Calculate the weights
    alpha[0] = 0.1 / (S[0] + vareps)**2
    alpha[1] = 0.6 / (S[1] + vareps)**2
    alpha[2] = 0.3 / (S[2] + vareps)**2
    
    omega[0] = alpha[0] / alpha.sum()
    omega[1] = alpha[1] / alpha.sum()
    omega[2] = alpha[2] / alpha.sum()
    
    # Calculate the left flux candidates
    qs[0] = (1.0/3.0)*qloc[0] - (7.0/6.0)*qloc[1] + (11./6.)*qloc[2]
    
    qs[1] = (-1.0/6.0)*qloc[1] + (5.0/6.0)*qloc[2] + (1./3.)*qloc[3]
    
    qs[2] = (1.0/3.0)*qloc[2] + (5.0/6.0)*qloc[3] - (1./6.)*qloc[4]
    
    retval = omega[0] * qs[0] + omega[1] * qs[1] + omega[2] * qs[2] 
    
    return retval
    
def get_right_flux(qloc):
    '''
    Note that qloc should be a stencil with 6 elements even though the first
    element will not be used in the calculation.
    '''
    if len(qloc) <> 6:
        raise ValueError('Stencil is not 6')
    
    vareps= 1e-6
    S = np.zeros(3)
    alpha = np.zeros(3)
    omega = np.zeros(3)
    qs = np.zeros(3)
    
    # Calculate smoothness measurements
    S[0] = (13./12.) * (qloc[1] - 2.0*qloc[2] + qloc[3])**2 + (1.0/4.0)* (qloc[1]-4.0*qloc[2] + 3.0*qloc[3])**2 
    
    S[1] = (13./12.) * (qloc[2] - 2.0*qloc[3] + qloc[4])**2 + (1.0/4.0)* (qloc[2] - qloc[4])**2 
    
    S[2] = (13./12.) * (qloc[3] - 2.0*qloc[4] + qloc[5])**2 + (1.0/4.0)* (3.0*qloc[3] - 4.0*qloc[4] + qloc[5])**2

    
    # Calculate the weights
    alpha[0] = 0.3 / (S[0] + vareps)**2
    alpha[1] = 0.6 / (S[1] + vareps)**2
    alpha[2] = 0.1 / (S[2] + vareps)**2
    
    omega[0] = alpha[0] / alpha.sum()
    omega[1] = alpha[1] / alpha.sum()
    omega[2] = alpha[2] / alpha.sum()
    
    # Calculate the left flux candidates
    qs[0] = (-1.0/6.0)*qloc[1] + (5.0/6.0)*qloc[2] + (1./3.)*qloc[3]
    
    qs[1] = (1.0/3.0)*qloc[2] + (5.0/6.0)*qloc[3] - (1./6.)*qloc[4]
    
    qs[2] = (11.0/6.0)*qloc[3] - (7.0/6.0)*qloc[4] + (1./3.)*qloc[5]
    
    retval = omega[0] * qs[0] + omega[1] * qs[1] + omega[2] * qs[2] 
    
    return retval
     
'''
PG - Note that the WENO and ENO functions below were copied from a question on scicomp.stackexchange

http://scicomp.stackexchange.com/questions/20054/implementation-of-1d-advection-in-python-using-weno-and-eno-schemes
'''    

def ENOweights(k,r):
    #Purpose: compute weights c_rk in ENO expansion 
    # v_[i+1/2] = \sum_[j=0]^[k-1] c_[rj] v_[i-r+j]
    #where k = order and r = shift 

    c = np.zeros(k)

    for j in range(0,k):
            de3 = 0.
            for m in range(j+1,k+1):
                #compute denominator 
                de2 = 0.
                for l in range(0,k+1):
                    #print 'de2:',de2
                    if l is not m:
                        de1 = 1.
                        for q in range(0,k+1):
                            #print 'de1:',de1
                            if (q is not m) and (q is not l):
                                de1 = de1*(r-q+1)


                        de2 = de2 + de1


                #compute numerator 
                de1 = 1.
                for l in range(0,k+1):
                    if (l is not m):
                        de1 = de1*(m-l)

                de3 = de3 + de2/de1


            c[j] = de3


    return c

def nddp(X,Y):
    #Newton's divided difference table 
    #the input are two vectors X and Y that represent points 

    n = len(X)

    DD = np.zeros((n,n+1))

    #inserting x into 1st column of DD-table 
    DD[:,0]=X

    #inserting y into 2nd column of DD-table
    DD[:,1]=Y

    #creates divided difference coefficients 
    #e.g: D[0,0] = (Y[1]-Y[0])/(X[1]-X[0])

    for j in range(0,n-1):
        for k in range(0,n-j-1): #j goes from 0 to n-2
            DD[k,j+2]= (DD[k+1,j+1]-DD[k,j+1])/(DD[k+j+1,0]-DD[k,0])

    return DD

def ENO(xloc, uloc, k):
    #Purpose: compute the left and right cell interface values using an ENO 
    #Approach based on 2k-1 long vectors uloc with cell k 

    #treat special case of k=1 - no stencil to select 
    if (k==1):
        ul = uloc[0]
        ur = uloc[0]

    #Apply ENO procedure 
    S = np.zeros(k,dtype=int)
    S[0] = k
    for kk in range (0,k-1):
        #print 'S:',S
        #left stencil
        xvec = np.zeros(k)
        uvec = np.zeros(k)
        Sindxl = np.append(S[0]-1, S[0:kk+1])-1
        xvec = xloc[Sindxl]
        uvec = uloc[Sindxl]
        DDl = nddp(xvec,uvec)
        Vl = abs(DDl[0,kk+2])

        #right stencil 
        xvec = np.zeros(k)
        uvec = np.zeros(k)
        Sindxr = np.append(S[0:kk+1], S[kk]+1)-1
        xvec = xloc[Sindxr]
        uvec = uloc[Sindxr]
        DDr = nddp(xvec,uvec)
        Vr = abs(DDr[0,kk+2])

        #choose stencil through divided differences 
        if (Vr>Vl):
            #print 'Vr>Vl'
            S[0:kk+2] = Sindxl+1
        else:
            S[0:kk+2] = Sindxr+1

    #Compute stencil shift 'r'
    r = k - S[0]

    #Compute weights for stencil 
    cr = ENOweights(k,r)
    cl = ENOweights(k,r-1)

    #Compute cell interface values 
    ur = 0 
    ul = 0 
    for i in range(0,k):
        ur = ur + cr[i]*uloc[S[i]-1]
        ul = ul + cl[i]*uloc[S[i]-1]

    return (ul,ur)

def WENO(xloc, uloc, k):
    #Purpose: compute the left and right cell interface values using ENO 
    #approach based on 2k-1 long vectors uloc with cell k 

    #treat special case of k = 1 no stencil to select 
    if (k==1):
        ul = uloc[0]
        ur = uloc[1]

    #Apply WENO procedure 
    alphal = np.zeros(k)
    alphar = np.zeros(k)
    omegal = np.zeros(k)
    omegar = np.zeros(k)
    betal = np.zeros(k)
    betar = np.zeros(k)
    d = np.zeros(k)
    vareps= 1e-6

    #Compute k values of xl and xr based on different stencils 
    ulr = np.zeros(k)
    urr = np.zeros(k)

    # These are the q values
    # Eq. 20, 21, 22
    for r in  range(0,k):
        cr = ENOweights(k,r)
        cl = ENOweights(k,r-1)
        
        for i in range(0,k):
            #urr[r] = urr[r] + cr[i]*uloc[k-r+i-1] 
            urr[r] = urr[r] + cr[i]*uloc[k-r+i]
            ulr[r] = ulr[r] + cl[i]*uloc[k-r+i-1] 


    #setup WENO coefficients for different orders -2k-1
    if (k==2):
        d[0]=2/3.
        d[1]=1/3.
        beta[0] = (uloc[2]-uloc[1])**2
        beta[1] = (uloc[1]-uloc[0])**2


    if(k==3):
        # These are the parameters used in Eqs 26-28
        d[0] = 3/10. 
        d[1] = 3/5.
        d[2] = 1/10.
        # These are the S values
        # Eqs. 29-31 in reverse order
        betal[0] = 13/12.*(uloc[2]-2*uloc[3]+uloc[4])**2 + 1/4.*(3*uloc[2]-4*uloc[3]+uloc[4])**2
        betal[1] = 13/12.*(uloc[1]-2*uloc[2]+uloc[3])**2 + 1/4.*(uloc[1]-uloc[3])**2
        betal[2] = 13/12.*(uloc[0]-2*uloc[1]+uloc[2])**2 + 1/4.*(3*uloc[2]-4*uloc[1]+uloc[0])**2
        
        betar[0] = 13/12.*(uloc[3]-2*uloc[4]+uloc[5])**2 + 1/4.*(3*uloc[3]-4*uloc[4]+uloc[5])**2
        betar[1] = 13/12.*(uloc[2]-2*uloc[3]+uloc[4])**2 + 1/4.*(uloc[2]-uloc[4])**2
        betar[2] = 13/12.*(uloc[1]-2*uloc[2]+uloc[3])**2 + 1/4.*(3*uloc[3]-4*uloc[2]+uloc[1])**2

    #compute alpha parameters 
    # Eq. 26, 27, 28 - likely in reverse order
    # These are the alpha values
    for r in range(0,k):
        alphar[r] = d[r]/(vareps+betar[r])**2
        alphal[r] = d[k-r-1]/(vareps+betal[r])**2

    #Compute WENO weights parameters 
    # These are the omega values
    # EQ 23,24,25
    for r in range(0,k):
        omegal[r] = alphal[r]/alphal.sum()
        omegar[r] = alphar[r]/alphar.sum()

    #Compute cell interface values
    # Eq 18
    ul = 0 
    ur = 0 
    for r in range(0,k):
        ul = ul + omegal[r]*ulr[r]
        ur = ur + omegar[r]*urr[r]

    return (ul,ur)
    
    
    
def WENO_original(xloc, uloc, k):
    '''
    This is the original version extracted from the website.
    
    '''
    
    #Purpose: compute the left and right cell interface values using ENO 
    #approach based on 2k-1 long vectors uloc with cell k 

    #treat special case of k = 1 no stencil to select 
    if (k==1):
        ul = uloc[0]
        ur = uloc[1]

    #Apply WENO procedure 
    alphal = zeros(k)
    alphar = zeros(k)
    omegal = zeros(k)
    omegar = zeros(k)
    beta = zeros(k)
    d = zeros(k)
    vareps= 1e-6

    #Compute k values of xl and xr based on different stencils 
    ulr = zeros(k)
    urr = zeros(k)

    for r in  range(0,k):
        cr = ENOweights(k,r)
        cl = ENOweights(k,r-1)

        for i in range(0,k):
            urr[r] = urr[r] + cr[i]*uloc[k-r+i-1] 
            ulr[r] = ulr[r] + cl[i]*uloc[k-r+i-1] 


    #setup WENO coefficients for different orders -2k-1
    if (k==2):
        d[0]=2/3.
        d[1]=1/3.
        beta[0] = (uloc[2]-uloc[1])**2
        beta[1] = (uloc[1]-uloc[0])**2


    if(k==3):
        d[0] = 3/10. 
        d[1] = 3/5.
        d[2] = 1/10.
        beta[0] = 13/12.*(uloc[2]-2*uloc[3]+uloc[4])**2 + 1/4.*(3*uloc[2]-4*uloc[3]+uloc[4])**2
        beta[1] = 13/12.*(uloc[1]-2*uloc[2]+uloc[3])**2 + 1/4.*(uloc[1]-uloc[3])**2
        beta[2] = 13/12.*(uloc[0]-2*uloc[1]+uloc[2])**2 + 1/4.*(3*uloc[2]-4*uloc[1]+uloc[0])**2

    #compute alpha parameters
    for r in range(0,k):
        alphar[r] = d[r]/(vareps+beta[r])**2
        alphal[r] = d[k-r-1]/(vareps+beta[r])**2

    #Compute WENO weights parameters
    for r in range(0,k):
        omegal[r] = alphal[r]/alphal.sum()
        omegar[r] = alphar[r]/alphar.sum()

    #Compute cell interface values
    ul = 0 
    ur = 0 
    for r in range(0,k):
        ul = ul + omegal[r]*ulr[r]
        ur = ur + omegar[r]*urr[r]

    return (ul,ur)