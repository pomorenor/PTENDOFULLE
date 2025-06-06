import numpy as np
import sympy as sy 
import itertools
import pandas as pd
import pywigxjpf as wig
from numpy import linalg as LA
import mpmath as mp 
from scipy.special import genlaguerre, gamma, factorial
from scipy.integrate import quad 
from math import sqrt, pi



def V(r):
       return 0.0174 + 0.0146*r**2 + 0.00174*r**4 + 0.000901*r**6 


Nu = 9.5e-4
beta = 2*pi*3*3*Nu


def Fnl(R,v,l):
        k = (v-l)//2
        laguerre_poly = genlaguerre(k,l+0.5)

        denom = (2**(k+l+1)*gamma(k+l+1.5))/sqrt(pi)

        prefactor = 2*sqrt((beta**((2*l +3)/2)*2**(k+l)*factorial(k))/denom)

        return prefactor*R**l*np.exp(-beta*R**2/2)*laguerre_poly(beta*R**2)

def radialIntegrand(r, n, l):
        R = Fnl(r,n,l)
        return R*V(r)*R*r**2

def radialMatrixElement(n,l):
        result, error = quad(radialIntegrand,0, np.inf, args=(n,l))
        return result


def compute_m(j):
	projection = [i for i in range(-j,j+1)]
	return projection

def couple_angular_momenta(l,j):
	coupled_angular_momenta = [i for i in range(np.abs(l-j),l+j+1)]
	return coupled_angular_momenta

#This functions form the possible couples for addition from the set of angular momenta A 
#and the set of angular momenta B, id. est, the AxB

def form_possible_momentum_couples(A,B):
	moment_couples_1 = itertools.product(A,B)
	
	all_couples = []
	
	for i in moment_couples_1:
		all_couples.append(i) 

	return list(set(all_couples))


def compute_ordered_basis_set(A, B, LAMBDA):
    basis = []
    for mLAMBDA in compute_m(LAMBDA):
        for l in A:
            for j in B:
                if LAMBDA in couple_angular_momenta(l, j):  # Only valid couplings
                    for ml in compute_m(l):
                        for mj in compute_m(j):
                            if (ml + mj) == mLAMBDA:
                                basis.append([l, ml, j, mj, LAMBDA, mLAMBDA])
    return basis


def kronecker_delta(i,j):

        if(i==j):
                return 1
        elif(i!=j):
                return 0

def compute_free_energies(J,k,n,cm_freq,vib_freq,B,C):

        K = np.abs(k)
        com_energy = (n+1.5)*cm_freq
        vibrational_energy = 0.0
 
        for i in vib_freq:
                vibrational_energy += 0.5*i

        rotational_energy = B*J*(J+1)+K**2*(C-B) 

        total_energy = com_energy + vibrational_energy + rotational_energy
        
        return total_energy


def compute_free_matrix_element(l,ml,j,mj,lamb,mlamb,lprime,mlprime,jprime,mjprime,lambprime,mlambprime,k,n,cm_freq,vib_freq,B,C):


        return compute_free_energies(j,k,n,cm_freq,vib_freq,B,C)*kronecker_delta(j,jprime)*kronecker_delta(l,lprime)*kronecker_delta(ml,mlprime)*kronecker_delta(mj,mjprime)*kronecker_delta(lamb,lambprime)*kronecker_delta(mlamb,mlambprime)

        


def compute_matrix_element(l,ml,j,mj,lamb,mlamb,lprime,mlprime,jprime,mjprime,lambprime,mlambprime,k,kprime,K,L,J,F,mF):
        coefficient1 = (-1)**(-J-j-jprime+mlamb+mlambprime+mF+mj+mlprime+k)
        coefficient2 = np.sqrt(1/(4*np.pi))
        coefficient3 = (2*lamb+1)*np.sqrt((2*lambprime+1)*(2*F+1)*(2*lamb+1)*(2*l+1)*(2*L+1)*(2*lprime+1)*(2*jprime+1)*(2*J+1)*(2*j+1))
        wigner1 = wig.wig3jj(2*l,2*L,2*lprime,0,0,0)
        wigner2 = wig.wig3jj(2*jprime, 2*J, 2*j,2*kprime, 2*K,2*k)
        wigner3 = wig.wig3jj(2*lambprime,2*F,2*lamb,2*mlambprime,2*mF,2*mlamb)
        wigner4 = wig.wig9jj(2*lprime,2*jprime,2*lambprime,2*L,2*J,2*F,2*l,2*j,2*lamb)
	 
        return coefficient1*coefficient2*coefficient3*wigner1*wigner2*wigner3*wigner4



"""
test = form_possible_momentum_couples([i for i in range(0,2)],[i for i in range(0,2)])

coupled_moments = [couple_angular_momenta(i[0],i[1]) for i in test]
coupled_moments_list = []

print(coupled_moments)

for i in coupled_moments:
	for j in range(0,len(i)):
		coupled_moments_list.append(i[j])

print(list(set(coupled_moments_list))	)

for i in coupled_moments_list:
	print(compute_m(i))
"""

n = 0

A = [0,2,4]
B = [0,1,3,4]
couples = form_possible_momentum_couples(A,B)
coupled_moments = [couple_angular_momenta(i[0],i[1]) for i in couples]
coupled_moments_list = []
for i in coupled_moments:
	for j in range(0,len(i)):
		coupled_moments_list.append(i[j])

non_repeated_moments = list(set(coupled_moments_list))

L = 2



print("Testing function for constructing the basis set")
BASIS = compute_ordered_basis_set(A,B,L)

print(BASIS)


wig.wig_table_init(20,9)
wig.wig_temp_init(20)

H_Matrix =  np.empty((len(BASIS),len(BASIS)))

#for i in BASIS:


#np.set_printoptions(precision=3, suppress=True)

ii=0
jj=0
for i in BASIS:
        for j in BASIS:
                #print(compute_matrix_element(i[0],i[1],i[2],i[3],i[4],i[5],j[0],j[1],j[2],j[3],j[4],j[5],0,0,0,0,0,0,0))
               # H_Matrix[BASIS.index(i),BASIS.index(j)] = compute_matrix_element(i[0],i[1],i[2],i[3],i[4],i[5],j[0],j[1],j[2],j[3],j[4],j[5],0,0,0,0,0,0,0)
                # H_Matrix[BASIS.index(i),BASIS.index(j)] = compute_free_matrix_element(i[0],i[1],i[2],i[3],i[4],i[5],j[0],j[1],j[2],j[3],j[4],j[5],0,0,271.0,[717.87,519.55,519.55],2.92,1.46)
                H_Matrix[BASIS.index(i),BASIS.index(j)] = (compute_free_matrix_element(i[0],i[1],i[2],i[3],i[4],i[5],j[0],j[1],j[2],j[3],j[4],j[5],0,0,271.0,[717.87,519.55,519.55],2.92,1.46)+radialMatrixElement(n,ii)*compute_matrix_element(i[0],i[1],i[2],i[3],i[4],i[5],j[0],j[1],j[2],j[3],j[4],j[5],0,0,0,0,0,0,0) )


                

df = pd.DataFrame(H_Matrix)
df.to_csv('Matrix.txt', header=None, index=None, sep=' ', mode='a')
print(df)



matrix = mp.matrix(H_Matrix)
eigenvalues, eigenvectors = mp.eig(matrix)

#print(eigenvalues)



print("The eigenvectors are: \n")

print(eigenvectors)






print("\n")
print("The eigenvalues are: \n")
#print(np.dot(eigenvectors[0],eigenvectors[1]))
#print(eigenvalues)
for eig in eigenvalues:
        print(eig.real)

print("\n")
print("Ender der Eigenwerte")
        


#print(compute_free_energies(0,0,0,271.0,[717.87,519.55,519.55],0,0))
#print(kronecker_delta(2,1))

