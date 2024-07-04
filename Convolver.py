import numpy as np
import matplotlib.pyplot as plt

def H(z, H0=70.1, Omega_m=0.276, Omega_Lambda=0.72391499):
    """
    H : Hubble parameter

    parameters
    z :  redshift
    H0 : Hubble constant or current value of Hubble parameter
    Omega_m : matter density parameter
    Omega_Lambda :  dark energy density parameter

    returns
    value of Hubble parameter at given redshift for given cosmological model
    """
    Omega_k = 1.0 - Omega_m - Omega_Lambda
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda + Omega_k * (1 + z)**2)



def mfp(z, H_z ,Lbox=None, Ncell=None):
    """
    mfp: Mean free path of ionizing photon at given redshift z (in Mpc)
    Normalized to the simulation scale proovided the length of the box and grid resolution
    
    parameters
    z : redshift
    H_z : Hubble parameter based on redshift
    Lbox : Length of simulation box (Grid spacing and initial simulation resolution taken)
    Ncell: Correponds to the resolution of the output
            (Not the initial resolution since the output is scaled down,
            and we are considering in terms of number of grids for mfp)

    returns
    lamda: mean free path of ionizing photon
    """
    c = 2.9979246*10**8 #m/s
    lamda = (c / (H_z*1000)) * 0.1 * np.power((1+z)/4, -2.55) # divided by 1000 to convert the distance in Mpc (km to m)
    if Ncell!=None:
        lamda = lamda*(Ncell/(Lbox*0.7)) #0.7 to convert the comoving  distance into physical distance
    return lamda

def file_reader(filename):
    """
    Reads halo / density maps as 3D numpy array
    """
    f = open(filename) #path to map
    N  = np.fromfile(f, count=3, dtype='uint64')
    N1,N2,N3 = N
    l = np.fromfile(f, count=1, dtype='float32')
    data1 = np.fromfile(f, count=N1*N2*N3, dtype='float32')
    f.close()
    data = np.reshape(data1, (N1,N2,N3), order='C') # row major order: going row-by-row
    return data

def spherical_window_3d(radius, N, normalization=False):
    """
    Spherical window function for smoothing the fields

    parameters
    radius: radius of the sphere used for smoothing aka mean free path
    N: Ncell
    Normalization: When set True, normalizes the spherical window function

    returns
    Noramlized spherical window function of given radius (mean free path scaled to grid numbers)
    """
    thw = np.zeros((N, N, N)) 
    origin = N // 2 -1  # to resolve the error of N==odd
    x = np.arange(-origin, -origin + N)
    y = np.arange(-origin, -origin + N)
    z = np.arange(-origin, -origin + N)

    # meshgrid to store 2D Array
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Calculating distance from origin
    grid_dist = np.sqrt(X**2 + Y**2 + Z**2)
    
    # if the grid point lies within the radius mark it
    thw[grid_dist <= radius] = 1.0

    # normalizing 
    if normalization==True:
        thw= thw/np.sum(thw)
    return thw


def transfomer(grid_3d, sphere_3d):
    """
    Follows convolution theorem
    Finds the convolution of the given field with filter function or the window function
    """

    # fourier transform of the the given field 
    fft_grid=np.fft.fftn(grid_3d)                         # fourier transfrom
    fft_grid_shift=np.fft.fftshift(fft_grid)           # shifting the center to middle of the grid

    # fourier transform of spherical window function
    window_ft=np.fft.fftn(sphere_3d)
    window_ft_shift=np.fft.fftshift(window_ft)

    #product FT(window) * FT(gaussian random field)
    ft_product_shift=fft_grid_shift*window_ft_shift

    # Doing the inverse fourier transform of the product
    inv_ft_product=np.fft.ifftn(ft_product_shift)

    return inv_ft_product


def convolver(filename, Ncell, sphere_radius):
    """
    Combines all the operations for convolution of field with a spherical window function
    
    parameters
    filename: name of the file you want to do convolution of
    Ncell: Dimension of the 3D array aka resolution of output of the simulation
    sphere_radius: mean free path of ionizing photon, we can directly put our function here

    returns
    Smoothed field
    """
    return np.abs(transfomer(file_reader(filename), spherical_window_3d(sphere_radius,Ncell,True)))


def plotter(grid_3d, title, slice_index=16):
    plt.figure(figsize=(10,11.5))
    plt.imshow(grid_3d[:, :, slice_index], cmap='magma', origin='lower', extent=[0, grid_3d.shape[0], 0, grid_3d.shape[1]])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.title(title)
    plt.show()

def fld_rdr(filename):
    """
    fld_rdr: field reader
    Used for reading the obtained ionized hydrogen fields
    """
    f = open(filename)
    N = np.fromfile(f, count=3, dtype='uint32')
    N1, N2, N3 =  N
    data = np.fromfile(f, count=N1*N2*N3, dtype='float32')
    f.close()
    return np.reshape(data, (N1, N2, N3), order='C')