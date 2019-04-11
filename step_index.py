#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:27:05 2019

@author: kalosu
"""

import matplotlib.pyplot as plt
from numpy import *
from scipy.special import *
#Definition for computing the corresponding normalized frequency
def norm_freq(a,wavelength,n_core,n_clad):
    k0 = (2*pi) / wavelength
    V = a * k0 * sqrt((n_core**2) - (n_clad**2))
    return V

#Definition for the eigen value equation - Hybrid modes
def eigen_HE(V,nu,u,wavelength,n_core,n_clad,a):
    k0=(2*pi) / wavelength
    w = sqrt(V**2 - u**2)
    betta = sqrt((k0*n_core)**2 - (u/a)**2)    
    eigen = ((jvp(nu,u,1) / (u*jv(nu,u))) + ( kvp(nu,w,1)/(w*kv(nu,w)))) * ((jvp(nu,u,1) / (u*jv(nu,u))) + ( kvp(nu,w,1)/(w*kv(nu,w)))*(n_clad**2/n_core**2)) - ( (((nu*betta)/(k0*n_core))**2 ) * (V/(u*w))**4)
    
    return eigen

def bessel_first_deriv(nu,u):
    deriv = jv(nu-1,u) - (nu/u)*jv(nu,u)
    return deriv

def bessel_mod_first_deriv(nu,w):
    deriv= - kv(nu-1,w) - (nu/w)*kv(nu,w)
    return deriv

#The following function is used to compute an approximation for the u value corresponding to the initial parameters. This is done in order to reduce the computation domain 
def u_approx(V,n_core,n_clad):
    delta = (n_core**2 - n_clad**2)/(2*n_core**2)
    u = 2.405*exp(-(1+delta)/V)
    return u
#Root finding algorithm
def root_finding(V,nu,wavelength,n_core,n_clad,a):
    blocks = 50
    ini_val = 1
    approx_u_val = u_approx(V,n_core,n_clad)
    #Dividing the domain in sub blocks in order to verify the existance of a 0 within any specific sub block
    interval_points = linspace(ini_val,ceil(approx_u_val),blocks)
    flag=0
    i=0
    partial_ini_val = ini_val
    partial_fin_val = ceil(approx_u_val)
    #The algorithm is run until the desired accuracy is obtained
    while abs(partial_ini_val -partial_fin_val)>0.000000000001:
        i=0
        flag=0
        while flag ==0 and i<=blocks-2:
            #If the sign of the sub block extreme function evaluation's values are different, these two extrema are stored for defining a new computation domain
            if sign(eigen_HE(V,nu,interval_points[i],wavelength,n_core,n_clad,a)) != sign(eigen_HE(V,nu,interval_points[i+1],wavelength,n_core,n_clad,a)):
                flag=1
                partial_ini_val = interval_points[i]
                partial_fin_val = interval_points[i+1]
                partial_func_ini_val = eigen_HE(V,nu,interval_points[i],wavelength,n_core,n_clad,a)
                partial_func_fin_val = eigen_HE(V,nu,interval_points[i+1],wavelength,n_core,n_clad,a)
            i=i+1
            #Defining new computation domain using the values obtained from above if statement
        interval_points = linspace(partial_ini_val,partial_fin_val,blocks)
    return partial_ini_val

#The following function is used for computing the value of the propagation constant (betta) and the effective refractive index 
def betta_neff_value(V,a,n_core,n_clad,u,wavelength):
    k0 = (2*pi) / wavelength
    delta = (n_core**2 - n_clad**2)/(2*n_core**2)
    betta = (1/a)*sqrt(((V**2)/(2*delta)) - u**2)
    neff = betta / k0 
    return betta,neff


if __name__ == "__main__":
    a = 5 #Core radius [um]
    wavelength = 1 #Free space wavelength [um]
    n_core = 1.55 #Core refractive index
    n_clad = 1.46 #Cladding refractive index
    nu = 1 #Bessel function order 
    V = norm_freq(a,wavelength,n_core,n_clad) #Computing the required normalized frequency
    #vector_test = linspace(1,V-1)
    #value_eigen_HE = eigen_HE(V,nu,vector_test,wavelength,n_core,n_clad,a)
    #plt.plot(vector_test,value_eigen_HE)
    #plt.show()
    u = root_finding(V,nu,wavelength,n_core,n_clad,a) #Computing the corresponding transversal phase constant U (Core parameter)
    betta,neff = betta_neff_value(V,a,n_core,n_clad,u,wavelength) #Computing the corresponding effective refractive index and propagation constant
    print ("The computed propagation constant (betta) is: %.15f:" % betta)
    print ("The computed effective refractive index is: %.15f" % neff)

