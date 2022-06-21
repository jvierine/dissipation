import h5py
import matplotlib.pyplot as plt
import numpy as n

import scipy.optimize as sio

def fit_dissipation_rate(s_h,R,err,epsilon0=10e-3,plot=False,c_k_p=0.5,c_k_std=0.1):
    print(R[0])
    s_h=s_h*1e3
    
    # forward model
    def model(x):
        R0=x[0]
        eps0=x[1]
        c_k=x[2]                
        S = c_k*eps0**(2.0/3.0)*s_h**(2.0/3.0)
        # acf
        Rm = R0 - S/2.0
        return(Rm)

    def ss(x):
        R_model=model(x)
        weighted_sum= n.sum(n.abs(R - R_model)**2.0/err**2.0) - n.log( (1.0/n.sqrt(0.3**2.0))*n.exp(-0.5*(x[2]-c_k_p)**2.0/c_k_std**2.0))
        return(weighted_sum)

    x0=n.array([R[0],10e-3,c_k_p])
    xhat=sio.fmin(ss,x0)
    Rm=model(xhat)
    print(xhat)
    if plot:
        plt.plot(s_h,Rm)
        plt.plot(s_h,R,"x")
        plt.show()
    return(xhat)

    
def estimate_dissipation_rate(R,s_h):
    plt.plot(R,s_h)
    plt.show()

if __name__ == "__main__":
    
    # Read data for longitudinal and transverse wind velocity correlation functions
    h=h5py.File("Rlltt.h5","r")

    # Kolmogorov constant for inertial subrange in the
    # case of stratified turbulence (Riley and Lindborg, 2008)
    c_k=2.0

    # correlation function
    # one for each day of the year
    R=h["R"][()]
    # copy R
    RO=n.copy(h["R"][()])
    
    # horizontal lags in km
    s_h=h["s_h"][()]    

    # number of days
    n_days=R.shape[0]

    # number of lags
    n_lags=R.shape[1]

    epsl=n.zeros(365)
    epst=n.zeros(365)    
    wl=3
    for i in range(365):

        # skip first lag
        # mean of longitudinal CF
        Rlf=n.nanmean(R[ n.max([i-wl,0]):n.min([i+wl,365]), 1:R.shape[1], 0],axis=0)
        # mean of transverse CF        
        Rtf=n.nanmean(R[ n.max([i-wl,0]):n.min([i+wl,365]), 1:R.shape[1], 1],axis=0)

        # estimate errors
        err_l=n.nanstd(R[n.max([i-wl,0]):n.min([i+wl,365]),:,0],axis=0)
        err_t=n.nanstd(R[n.max([i-wl,0]):n.min([i+wl,365]),:,1],axis=0)        

        err_l_f=err_l[1:len(err_l)]
        err_t_f=err_t[1:len(err_t)]        

        # fit dissipation rate
        s_h_f=s_h[1:R.shape[1]]
        xhat=fit_dissipation_rate(s_h_f, Rlf, err_l_f, c_k_p=c_k, plot=False)        
        epsl[i]=xhat[1]
        xhat=fit_dissipation_rate(s_h_f, Rtf, err_t_f, c_k_p=c_k, plot=False)
        epst[i]=xhat[1]        

    epst=n.array(epst)
    epsl=n.array(epsl)    
    plt.plot(epst*1e3)
    plt.plot(epsl*1e3)    
    plt.xlabel("Day of year")
    plt.ylabel("Dissipation rate (mW/kg)")
    plt.title("$C_k=%1.1f$"%(c_k))
    plt.show()

        
 #       plt.show()


    
