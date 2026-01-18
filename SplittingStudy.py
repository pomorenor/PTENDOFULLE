import numpy as np
import sympy as sy 
import itertools
import pandas as pd
import pywigxjpf as wig
from numpy import linalg as LA
import mpmath as mp 
from scipy.special import genlaguerre, gamma, factorial, factorial2
from scipy.integrate import quad 
from math import sqrt, pi



Nu = 1.235E-3
beta = 2*pi*3*5494.8926*Nu

#GQ,m_Q,L,J



#Obtain free energies in Eh
def compute_H0(J,k,n,cm_freq,vib_freq,B,C):

        K = np.abs(k)
        com_energy = (n+1.5)*cm_freq
        vibrational_energy = 0.0
 
        for i in vib_freq:
                vibrational_energy += 0.5*i

        rotational_energy = B*J*(J+1)+K**2*(C-B) 

        total_energy = com_energy + vibrational_energy + rotational_energy
        
        return total_energy*0.0000046



def G0000(r):

       return 0.513 + 0.43*r**2 + 0.0519*r**4 + 0.000901*r**6 
       #return 0.683 + 0.58*r**2 + 0.0646*r**4 + 0.036*r**6 
       #return 0.683 + 0.58*r**2 + 0.0646*r**4 + 0.036*r**6 

def G2002(r): 
       return 0.683 + 0.58*r**2 + 0.0646*r**4 + 0.036*r**6 


G = {(0,0,0,0):G0000,(2,0,0,2):G2002 }

def Fnl(R,n,l):
        k = (n-l)//2
        laguerre_poly = genlaguerre(k,l+0.5)

        prefactor = 2*sqrt((beta**((2*l +3)/2)*2**(k+l)*factorial(k))/np.sqrt(pi)*factorial2(2*(k+l)+1))

        return prefactor*R**l*np.exp(-beta*R**2/2)*laguerre_poly(beta*R**2)

def radialIntegrand(r, n, l, G):
        R = Fnl(r,n,l)
        return R*G(r)*R*r**2

def radialMatrixElement(n,l,G):
        result, error = quad(radialIntegrand,0, np.inf, args=(n,l,G))
        return result



def angularMatrixElement(l,j,lamb,L,J,Q,lprime,jprime, lambprime,kj,kJ,kjprime,mlamb, mQ, mlambprime):
        coefficient1 = (-1)**(kjprime)*(-1)**(-jprime)*(-1)**(-J+mQ)*(-1)**(-j+mlamb)
        coefficient2 = np.sqrt(1/((4*np.pi)*(8*np.pi**2)))
        coefficient3 = (2*lambprime+1)*np.sqrt((2*lambprime+1)*(2*Q+1)*(2*lamb+1)*(2*l+1)*(2*L+1)*(2*lprime+1)*(2*jprime+1)*(2*J+1)*(2*j+1))
        wigner1 = wig.wig3jj(2*lprime,2*L,2*l,0,0,0)
        wigner2 = wig.wig3jj(2*j, 2*J, 2*jprime,2*kj, 2*kJ,-2*kjprime)
        wigner3 = wig.wig3jj(2*lamb,2*Q,2*lambprime,2*mlamb,2*mQ,2*mlambprime)
        wigner4 = wig.wig9jj(2*l,2*j,2*lamb,2*L,2*J,2*Q,2*lprime,2*jprime,2*lambprime)
	 
        return coefficient1*coefficient2*coefficient3*wigner1*wigner2*wigner3*wigner4

def totalMatrixElement(G,Q,L,J,mQ,n,cm_freq,vib_freq,B,C,l,j,lamb,lprime,jprime, lambprime,kj,kJ,kjprime,mlamb,mlambprime ):

        result = 0.0

        for ll in L:
            for jj in J:
                for qq in Q:
                    for mQmQ in mQ:
                        key = (qq,mQmQ,ll,jj)
                        if key in G:  
                            result += angularMatrixElement(l,j,lamb,ll,jj,qq,lprime,jprime, lambprime,kj,kJ,kjprime,mlamb, mQmQ, mlambprime)*radialMatrixElement(n,l,G[key])

        if j== jprime and kj==kjprime:
               result += compute_H0(j,kj,n,cm_freq,vib_freq,B,C)
               
        return result

wig.wig_table_init(20,9)
wig.wig_temp_init(20)

print(totalMatrixElement(G, [0,2], [0], [0,2], [0], 0, 271.33, [717.87,519.55,519.55], 2.92, 1.46,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) #     def totalMatrixElement(G,Q,L,J,mQ,n,cm_freq,vib_freq,B,C,l,j,lamb,lprime,jprime, lambprime,kj,kJ,kjprime,mlamb,mlambprime ):
