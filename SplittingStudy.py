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
import matplotlib.pyplot as plt 


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

       return (0.513 + 0.43*r**2 + 0.0519*r**4 + 0.000901*r**6) 
       #return 0.683 + 0.58*r**2 + 0.0646*r**4 + 0.036*r**6 

def G2002(r): 
       return (0.683 + 0.58*r**2 + 0.0646*r**4 + 0.036*r**6) 

def G4004(r):
       return (-0.0469076 - 0.0361674*r**2 - 0.00615314*r**4 - 0.00214876*r**6)



G = {(0,0,0,0):G0000,(2,0,0,2):G2002, (4,0,0,4):G4004 }

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
        #coefficient1 = (-1)**(lprime+jprime+lamb+Q)
        coefficient1 = (-1)**(kjprime-jprime-J+mQ-j+mlamb)

        coefficient2 = np.sqrt(1/(32*np.pi**3))
        coefficient3 = (2*lambprime+1)*np.sqrt((2*lambprime+1)*(2*Q+1)*(2*lamb+1)*(2*l+1)*(2*L+1)*(2*lprime+1)*(2*jprime+1)*(2*J+1)*(2*j+1))
        wigner1 = wig.wig3jj(2*lprime,2*L,2*l,0,0,0)
        wigner2 = wig.wig3jj(2*j, 2*J, 2*jprime,2*kj, 2*kJ,-2*kjprime)
        wigner3 = wig.wig3jj(2*lamb,2*Q,2*lambprime,2*mlamb,2*mQ,2*mlambprime)
        wigner4 = wig.wig9jj(2*l,2*j,2*lamb,2*L,2*J,2*Q,2*lprime,2*jprime,2*lambprime)
	 
        return coefficient1*coefficient2*coefficient3*wigner1*wigner2*wigner3*wigner4



def build_basis_with_different_jk(jk_pairs):
    """
    Build basis with n=0, l=0 fixed, but different j,k values
    States: (j,k) = (5,2), (5,4), (4,1), (4,2), (3,1), (3,2), (2,2)
    For l=0, λ must equal j (since |l-j| ≤ λ ≤ l+j, so λ = j)
    mλ can range from -λ to λ, but let's include all mλ for completeness
    """
    n_val = 0
    l_val = 0
    
    # List of (j, k) pairs
    
    
    basis = []
    
    for j_val, kj_val in jk_pairs:
        # For l=0, λ must be j (since |0-j| ≤ λ ≤ 0+j)
        lamb = j_val
        
        # Include all mλ values from -λ to λ
        for mlamb in range(-lamb, lamb + 1):
            basis.append((n_val, l_val, j_val, lamb, mlamb, kj_val))
    
    return basis



wig.wig_table_init(20,9)
wig.wig_temp_init(20)

#print(220000*totalMatrixElement(G, [0,2], [0], [0,2], [0], 0, 271.33, [717.87,519.55,519.55], 2.92, 1.46,
           #             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)) #     def totalMatrixElement(G,Q,L,J,mQ,n,cm_freq,vib_freq,B,C,l,j,lamb,lprime,jprime, lambprime,kj,kJ,kjprime,mlamb,mlambprime ):


#print(build_basis_with_different_jk([(2,2)]))




## Fucking AI

def build_hamiltonian_matrix(basis, G_dict, Q_range, L_range, J_range, mQ_range,
                            cm_freq, vib_freq, B, C):
    """
    Build Hamiltonian matrix for given basis
    """
    N = len(basis)
    H = np.zeros((N, N), dtype=complex)
    
    print(f"Building {N}x{N} Hamiltonian matrix...")
    
    for i in range(N):
        state_i = basis[i]
        n_i, l_i, j_i, lamb_i, mlamb_i, kj_i = state_i
        
        for j in range(i, N):  # Build upper triangle
            state_j = basis[j]
            n_j, l_j, j_j, lamb_j, mlamb_j, kj_j = state_j
            
            # Initialize matrix element
            H_ij = 0.0
            
            # Calculate potential matrix element
            kJ_val = kj_j - kj_i  # From 3j symbol constraint
            
            # Sum over Q, L, J, mQ
            for ll in L_range:
                for jj in J_range:
                    for qq in Q_range:
                        for mQmQ in mQ_range:
                            key = (qq, mQmQ, ll, jj)
                            if key in G_dict:
                                # Angular part
                                angular = angularMatrixElement(
                                    l_i, j_i, lamb_i, ll, jj, qq,
                                    l_j, j_j, lamb_j,
                                    kj_i, kJ_val, kj_j, mlamb_i, mQmQ, mlamb_j
                                )
                                # Radial part (same for all since n=0, l=0)
                                radial = radialMatrixElement(n_i, l_i, G_dict[key])
                                H_ij += angular * radial
            
            # Add free Hamiltonian if diagonal
            if i == j:
                H_ij += compute_H0(j_i, kj_i, n_i, cm_freq, vib_freq, B, C)
            
            H[i, j] = H_ij*220000 
            #if i != j:
            #    H[j, i] = np.conj(H_ij)  # Hermitian conjugate
    
    return H

def print_hamiltonian_matrix(H, basis):
    """
    Print Hamiltonian matrix with state labels
    """
    N = len(basis)
    
    print(f"\n{'='*70}")
    print(f"HAMILTONIAN MATRIX ({N}x{N})")
    print(f"{'='*70}")
    
    # Print basis states
    print("\nBASIS STATES:")
    print("Idx | n | l | j | λ | mλ | kj")
    print("-" * 40)
    for idx, state in enumerate(basis):
        n, l, j, lamb, mlamb, kj = state
        print(f"{idx:3d} | {n} | {l} | {j} | {lamb} | {mlamb:3d} | {kj}")
    
    # Print matrix (first 10x10 if large)
    print(f"\nMATRIX ELEMENTS (first {min(10, N)}x{min(10, N)} block):")
    print("-" * 70)
    
    size = min(10, N)
    # Column headers
    print(" " * 6, end="")
    for j in range(size):
        print(f"{j:12d}", end="")
    print()
    
    # Matrix rows
    for i in range(size):
        print(f"{i:4d}: ", end="")
        for j in range(size):
            val = H[i, j].real
            print(f"{val:12.6e}", end="")
        print()
    
    if N > size:
        print(f"... (showing first {size}x{size} of {N}x{N} matrix)")
    
    # Print all diagonal elements
    print(f"\nDIAGONAL ELEMENTS:")
    print("-" * 70)
    for i in range(N):
        val = H[i, i].real
        n, l, j, lamb, mlamb, kj = basis[i]
        print(f"H[{i},{i}] = {val:12.6e} Eh  (|j={j},k={kj},mλ={mlamb}⟩)")


# Now let's use the functions
if __name__ == "__main__":
    # Initialize Wigner library (already done)
    
    # Physical parameters
    cm_freq = 254.74
    vib_freq = [717.87, 519.55, 519.55]
    B = 2.92
    C = 1.46
    
    # Define the jk pairs you want
    jk_pairs = [(3,0)]
    # Build basis
    basis = build_basis_with_different_jk(jk_pairs)
    print(f"\nBasis has {len(basis)} states")
    
    # Summation ranges
    Q_range = [0]
    L_range = [0]
    J_range = [0]
    mQ_range = [0]
    
    # Build Hamiltonian matrix
    H = build_hamiltonian_matrix(basis, G, Q_range, L_range, J_range, mQ_range,
                                cm_freq, vib_freq, B, C)
    
    # Print the matrix
    print_hamiltonian_matrix(H, basis)
    
    # Optional: Diagonalize and print eigenvalues
    print(f"\n{'='*70}")
    print("EIGENVALUES:")
    print(f"{'='*70}")
    
    # Diagonalize (use real part)
    H_to_diag  = mp.matrix(H)
    eigenvalues, eigenvectors = mp.eig(H_to_diag)
    
    real_eigs = []
    for ii in eigenvalues:
        real_eigs.append(ii.real)
    #for i, eig in enumerate(eigenvalues):
    #    print(f"E{i+1:2d} = {eig:18.12f} Eh")
    
    for ii in real_eigs:
         #if (ii > 0.02248*22000 and ii<0.1*22000):
        print((ii)*0.0000046)

    #for ii in real_eigs:
        #print((ii)*0.0000046+0.016446573767559227)

    # Your eigenvalues list (real_eigs) from the code above
    # Convert to forrelative energies
    releg = [0.022512254,
0.02257388,
0.022567083,
0.022546693,
0.022726409,
0.022719612,
0.022699222,
0.022665239,
0.022906124,
0.022899328,
0.022878938,
0.022844955,
0.022797379,
0.023113026,
0.023106229,
0.02308584
]
    relative_energies = [(ii-0.022512254)*220000 for ii in releg]
    # Create vertical energy level plot
  
"""
    plt.figure(figsize=(8, 10))
    plt.hlines(y=relative_energies, xmin=0, xmax=1, colors='blue', linewidth=2)
    plt.scatter([0.5]*len(relative_energies), relative_energies, s=50, color='red')

    # Add labels and formatting
    plt.xlabel('Energy Levels')
    plt.ylabel('Relative Energy (Eh)')
    plt.title('Energy Level Diagram')
    plt.grid(True, axis='y', alpha=0.3)

    # Show the plot
    plt.show()
"""

print("Well bottom:")
print(angularMatrixElement(0,0,0,0,0,0,0,0, 0,0,0,0,0, 0, 0)*radialMatrixElement(0,0,G0000))
print("Total energy Ground State:")
print("cm-1:", 220000*(compute_H0(1,1,0,254.73,[717.87,519.55,519.55],2.92,1.46) + angularMatrixElement(0,1,0,0,0,0,0,1, 0,1,0,1,0, 0, 0)*radialMatrixElement(0,0,G0000)))
print("Eh:", 1*(compute_H0(1,1,0,254.73,[717.87,519.55,519.55],2.92,1.46) + angularMatrixElement(0,1,0,0,0,0,0,1, 0,1,0,1,0, 0, 0)*radialMatrixElement(0,0,G0000)))


print("State correction:")
print("cm-1:", 220000*(compute_H0(3,1,0,254.73,[717.87,519.55,519.55],2.92,1.46) + 0*angularMatrixElement(0,0,0,0,0,0,0,0, 0,0,0,0,0, 0, 0)*radialMatrixElement(0,0,G0000)))
print("Eh:", 1*(compute_H0(3,1,0,254.73,[717.87,519.55,519.55],2.92,1.46) + 0*angularMatrixElement(0,0,0,0,0,0,0,0, 0,0,0,0,0, 0, 0)*radialMatrixElement(0,0,G0000)))

print(relative_energies)



