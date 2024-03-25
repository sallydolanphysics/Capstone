import numpy as np

N_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
strength_list = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]

E = np.arange(0.001, 10, 0.005)
n_configs = 10000
L = 100

# Constants
h_bar = 1.0  # Planck's constant over 2π
m = 1.0    # Mass of the electron

def Mdelta(k, strength):
    L = strength * m

    Mdelta = np.array([ [0. + 0j, 0.+0j], [0.+0j, 0.+0j]])

    b = 2

    Mdelta[0,0] = 1.0 + L /(b*1j*k)
    Mdelta[0,1] = + L /(b*1j*k)
    Mdelta[1,0] = - L /(b*1j*k)
    Mdelta[1,1] = 1.0 - L/(b*1j*k) 

    return Mdelta


def M0(k, L):

    M0 = np.array([ [0. + 0j, 0.+0j], [0.+0j, 0.+0j]])

    M0[0,0] = np.exp(+1j*k*L)
    M0[0,1] = 0.0
    M0[1,0] = 0.0
    M0[1,1] = np.exp(-1j*k*L)

    return M0

def transfer_matrix_delta_potential(x, heights, k):
    M = np.identity(2, dtype=complex)  # Start with the identity matrix

    # Ensure x and heights are sorted in the same order
    indices = np.argsort(x)
    x_sorted = np.array(x)[indices]
    heights_sorted = np.array(heights)[indices]

    # Build the transfer matrix
    for i in range(len(x_sorted)):
        if i > 0:  # For all but the first potential, include free space evolution
            M = np.matmul(M0(k, x_sorted[i] - x_sorted[i-1]), M)
        M = np.matmul(Mdelta(k, heights_sorted[i]), M)

    M = np.identity(2, dtype=complex)

    for i in range(len(x)-1, -1, -1): #for each scatterer
        M = np.matmul(Mdelta(k, heights_sorted[i]), M) 
        if len(x)>1:
            M = np.matmul(M0(k, x[i]-x[i-1]), M) 

    return M

def trans_delta_L(x, heights, E):
    T = np.zeros(len(E))

    for i, energy in enumerate(E):
        k = np.sqrt( m * energy) / h_bar

        # Calculate the transfer matrix for given energy and delta potentials
        s = transfer_matrix_delta_potential(x, heights, k) # throwing error
        
        # Compute transmission coefficient
        T[i] = 1.0 / (np.abs(s[0, 0]))**2

    return T

for N in N_list:

    for strength in strength_list:

        strengths = strength*np.ones(N)
        sum_of_spectra = 0

        for config in range(n_configs):
            x = L*np.random.rand(N) 
            transmission_coefficients = trans_delta_L(x, strengths, E)
            sum_of_spectra = np.add(sum_of_spectra, transmission_coefficients)

        ca_T = sum_of_spectra / n_configs

        np.savetxt(f'ca_T_N_{N}_n_configs_{n_configs}_strength_{strength}_eoin_run', ca_T)

        print(f"done {strength}")