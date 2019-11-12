#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:14:04 2019

@author: Julio Candanedo
"""

import numpy as np
import scipy
import scipy.stats as sps
import matplotlib.pyplot as plt
import time

def au_temp(x):
    return x/315774.64 #
def Bohr(x): ### Given Ångström get Bohr
    return 1.88973*x
def eV(x):
    return 27.2114*x
def fs(x): ### Given time in a.u. get fs
    return x/41.341447 #
def Hartree(x): ### Given eV, give Ha
    return x/27.2114#
def number_density(ρ, mass_of_atom): ### ρ [g/cc]
    ρ = 0.089238919*ρ ### ρ [Daltons/å^3]
    ρ = 1822.88849*ρ ### ρ [m_e/å^3]
    return ρ/mass_of_atom
def Ångström(x): ### Given Bohr get Ångström
    return x/1.88973 #

######################## Simulation Parameters
Temperature = 5 ## in K
Temperature = au_temp(Temperature) ## a.u. temp
dt = 82.64 ## in a.u. = 2 fs
T = dt*30 ## in a.u. total simulation time
time_array = np.linspace(0., T, num=int(T/dt))
Total_CPU = 0

#### Input Beam Parameters
emittance_x = 20 ### emittance in Ångström-rad (SLAC UED is 2 - 20 nm-rad)
emittance_y = 20 ### emittance in Ångström-rad
energy_z = 3000000 ### forward beam energy in eV, highest energy i could find elastic CS in NIST
Γx = 56419 #4400 ## FWHM in Ångströms (Beam Radius), (SLAC UED is 10 μm = 100,000 Å)
Γy = 56419 #4400 ## FWHM in Ångströms
Γt = T/2 ## FWHM in au_time
fluence = 10 # electrons/Ångström^2

#### Derived Beam Parameters
emittance_x = Bohr(emittance_x)
emittance_y = Bohr(emittance_y)
emittance = np.sqrt(emittance_x*emittance_y)
fluence = fluence/3.57106483 ### fluence in electrons/Bohr^2
Γx = Bohr(Γx)
Γy = Bohr(Γy)
energy_z = Hartree(energy_z)
λ = 1/np.sqrt(2*energy_z) ### reduced electron beam wavelength
spot_size = 2*np.sqrt(Γx*Γy) ### Beam Spot Diameter
beam_area = np.pi*Γx*Γy ### Beam Area
Number_beam_particles = int(beam_area*fluence)
β_beam = np.sqrt(1 - 1/((1 + energy_z*(1/137)**2)**2)) ## in c


Z_dictonary = np.array(['e ', 'H ', 'He', 
                        'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',
                        'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])
Z_mass = np.array([ 1.,   1837.,   7296.,  
                   12650., 16427.,  19705.,  21894.,  25533.,  29164.,  34631., 36785.,  
                   41908., 44305.,  49185.,  51195.,  56462.,  58441.,  64621.,  72820.,  
                   71271., 73057.,  81949.,  87256.,  92861.,  94782., 100145., 101799., 107428., 106990., 115837., 119180., 127097., 132396., 136574., 143955., 145656., 152754.,
                   155798., 159721., 162065., 166291., 169357., 174906., 176820., 184239., 187586., 193991., 196631., 204918., 209300., 216395., 221954., 232600., 231331., 239332., 
                   242270., 250331., 253208., 255415., 256859., 262937., 264318., 274089., 277013., 286649., 289702., 296219., 300649., 304894., 307947., 315441., 318945., 325367., 329848., 335119., 339434., 346768., 350390., 355616., 359048., 365656., 372561., 377702., 380947., 380983., 382806., 404681.,
                   406504., 411972., 413795., 422979., 421152., 433900., 432024., 444784., 442961., 450253., 450253., 457545., 459367., 468482., 470305., 472128., 477596., 486711., 492179., 490357., 492179., 492179., 506763., 512231., 512231., 519523., 521346., 526814., 526814., 534106., 534106., 535929.])
Z_α = np.array([0.000, 4.507, 1.384, 
                164.113, 37.740, 20.500, 11.300, 7.400, 5.300, 3.740, 3.661, 2.700,
                71.200, 57.800, 37.300, 25.000, 19.400, 14.600, 10.083,
                289.700, 160.800, 97.2, 63.4, 68.2, 60, 66.8, 62.65, 57.71, 51.10, 58.7, 38.8, 46.6, 39.43, 29.8, 26.24, 21.03, 17.075,
                319.8, 186, 163, 112, 97.9, 87.1, 80.4, 65, 11, 32., 45.9, 49.65, 62.1, 67.5, 42.55, 37., 24.6, 27.815,
                400.8, 268, 170.7, 191.7, 238.9, 183.6, 200.2, 156.6, 154.8, 176.1, 158.6, 157.2, 145.1, 217.3, 129.6, 147.1, 123.5, 83.7, 58., 68.1, 65.6, 57.8, 51.7, 44., 27.9, 33.91, 51., 56., 54.7, 46.8, 43., 34.2,
                316.8, 232., 203.3, 217., 154.4, 137., 150.5, 132.2, 131.2, 143.6, 125.3, 121.5, 117.5, 113.4, 109.4, 114., 42.5, 40.7, 38.4, 36.2, 34.2, 32.3, 30.6, 29., 29.9, 30.59, 57.98 ]) ## replace metals with their valancy polarizability!!!
Z_Eion = np.array([0., 0.500, 0.904, 
                   0.198, 0.343, 0.305, 0.414, 0.534, 0.500, 0.640, 0.792, 
                   0.189, 0.281, 0.220, 0.300, 0.385, 0.381, 0.477, 0.579,
                   0.160, 0.225, 0.241, 0.251, 0.248, 0.249, 0.273, 0.290, 0.290, 0.281, 0.284, 0.345, 0.220, 0.290, 0.360, 0.358, 0.434, 0.514,
                   0.154, 0.209, 0.228, 0.244, 0.248, 0.261, 0.268, 0.270, 0.274, 0.306, 0.278, 0.331, 0.213, 0.270, 0.316, 0.331, 0.384, 0.446, 
                   0.143, 0.192, 0.205, 0.204, 0.201, 0.203, 0.205, 0.207, 0.208, 0.226, 0.215, 0.218, 0.221, 0.224, 0.227, 0.230, 0.199, 0.251, 0.277, 0.289, 0.288, 0.310, 0.330, 0.329, 0.339, 0.384, 0.224, 0.273, 0.268, 0.309, 0.342, 0.395,
                   0.149, 0.194, 0.190, 0.232, 0.216, 0.228, 0.230, 0.221, 0.220, 0.220, 0.228, 0.231, 0.236, 0.239, 0.242, 0.244, 0.180, 0.220, 0.249, 0.287]) ### https://www.webelements.com/seaborgium/atoms.html
Z_valance = np.array([0., 1., 2., 
                   1., 2., 3., 4., 5., 6., 7., 8., 
                   1., 2., 3., 4., 5., 6., 7., 8.,
                   1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 6., 7., 8.,
                   1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 6., 7., 8.])
Z_a = np.array([0., 1., 2., 
                3., 4., 5., 6., 7., 8., 9., 1.780, 
                11., 12., 13., 14., 15., 16., 17., 1.180, 
                19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 0.900, 
                37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 0.807, 
                55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 80., 81., 82., 83., 84., 85., 86., 
                87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 110., 111., 112., 113., 114., 115., 116., 117., 118.])
Z_exciton = np.array([0., 1., 2., 
                3., 4., 5., 6., 7., 8., 9., 0.2471, 
                11., 12., 13., 14., 15., 16., 17., 0.2165, 
                19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 0.2025, 
                37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 53., 0.1899, 
                55., 56., 57., 58., 59., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 80., 81., 82., 83., 84., 85., 86., 
                87., 88., 89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 110., 111., 112., 113., 114., 115., 116., 117., 118.])


def energy(r, α, E):
    Rmin = 2.54*(np.einsum('i, j -> ij', α**(1/7), np.ones(α.shape)) + np.einsum('i, j -> ij', np.ones(α.shape), α**(1/7)))
    C6 = 3/2*np.einsum('i, j -> ij', α*E, α*E)/(np.repeat([E],len(E),axis=0) + np.transpose(np.repeat([E],len(E),axis=0))) 
    London_energy = (C6/(Rmin)**6)*(np.exp(-(12/Rmin)*(r-Rmin)) - 2*np.exp(-(6/Rmin)*(r-Rmin)))
    np.fill_diagonal(London_energy, 0)

    return np.einsum('ij -> ', London_energy)/2

def force(r, α, E):
    Rmin = 2.54*(np.einsum('i, j -> ij', α**(1/7), np.ones(α.shape)) + np.einsum('i, j -> ij', np.ones(α.shape), α**(1/7)))
    C6 = 3/2*np.einsum('i, j -> ij', α*E, α*E)/(np.repeat([E],len(E),axis=0) + np.transpose(np.repeat([E],len(E),axis=0)))
    return (C6/(Rmin)**6)*(12*np.exp(-(12/Rmin)*(r-Rmin)) - 2*6*np.exp(-(6/Rmin)*(r-Rmin)))/Rmin

def Coulomb_force(r, a, b, Z):    
    aa = np.sqrt(np.einsum('ia, jb -> ijab', a, a)/(np.einsum('ia, jb -> ijab', a, np.ones(a.shape)) + np.einsum('ia, jb -> ijab', np.ones(a.shape), a)))
    r = r + np.identity(len(Z))
    #### added the a factor in exp term!!!
    f_nn = np.einsum('i, j, ij -> ij', Z, Z, r**(-2))
    f_ee = -2/(np.sqrt(np.pi))*np.einsum('ijab, ia, jb, ijab, ij -> ij', aa, b, b, np.exp(-(np.einsum('ijab, ij -> ijab', aa, r))**2), r**(-1)) + np.einsum('ia, jb, ijab, ij -> ij', b, b, scipy.special.erf(np.einsum('ijab, ij -> ijab', aa, r)), r**(-2)) ### issue is with the 1/r and 1/r**2!!!
    f_ne = -2/(np.sqrt(np.pi))*np.einsum('ia, i, ja, ija, ij -> ij', a, Z, b, np.exp(-(np.einsum('ia, ij -> ija', a, r))**2), r**(-1)) + np.einsum('i, jb, ijb, ij -> ij', Z, b, scipy.special.erf(np.einsum('ib, ij -> ijb', a, r)), r**(-2))
    f_en = -2/(np.sqrt(np.pi))*np.einsum('ja, i, ja, ija, ij -> ij', a, Z, b, np.exp(-(np.einsum('ja, ij -> ija', a, r))**2), r**(-1)) + np.einsum('i, ja, ija, ij -> ij', Z, b, scipy.special.erf(np.einsum('ja, ij -> ija', a, r)), r**(-2))
    
    return  f_nn + f_ee + f_ne + f_en

def Coulomb_energy(r, a, b, Z):
    one = np.ones(r.shape)
    np.fill_diagonal(one, 0)
    
    aa = np.sqrt(np.einsum('ia, jb -> ijab', a, a)/(np.einsum('ia, jb -> ijab', a, np.ones(a.shape)) + np.einsum('ia, jb -> ijab', np.ones(a.shape), a)))
    
    E_nn = np.einsum('i, j, ij, ij -> ', Z, Z, (r + np.identity(len(Rx_old)))**(-1), one)/2
    E_ee = np.einsum('ia, jb, ijab, ij, ij -> ', b, b, scipy.special.erf(np.einsum('ijab, ij -> ijab', aa, r)), (r + np.identity(len(Rx_old)))**(-1), one)/2
    E_ne = np.einsum('j, ib, ijb, ij, ij -> ', Z, b, scipy.special.erf(np.einsum('ib, ij -> ijb', a, r)), (r + np.identity(len(Rx_old)))**(-1), one)/2
    E_en = np.einsum('i, ja, ija, ij, ij -> ', Z, b, scipy.special.erf(np.einsum('ja, ij -> ija', a, r)), (r + np.identity(len(Rx_old)))**(-1), one)/2
    
    return E_nn + E_ee + E_ne + E_en

def Get_basis_vectors(r, x):
    ee = -x/(r + np.identity(len(r)))
    np.fill_diagonal(ee, 0)
    return ee

def Get_RDF(R, rr): ### δr and r_cutoff in Ångström
    binz = Bohr(rr) ### in Bohr
    shell_vol = np.diag(np.repeat([4*np.pi*binz**3/3],len(binz),axis=0) - np.transpose(np.repeat([4*np.pi*binz**3/3],len(binz),axis=0)), k=1) ### shell_volume = 4*np.pi*binz**3/3
    his = np.histogram(R.flatten(), bins=binz) ### Make_t R-Matrix into 1D array, we should get len(R) number of 0s and then duplicates of the rest!
    RDF = his[0]/shell_vol/his[0][0]
    RDF[0] = 0
    return (1/nA)*RDF

def Get_RMSD(x0, y0, z0, xt, yt, zt):
    return np.sqrt(np.einsum('i->', (xt - x0)**2 + (yt - y0)**2 + (zt - z0)**2)/len(x0))

def Get_CoM(mass, x, y, z):
    total_mass = np.einsum('i->', mass)
    CoMx = np.einsum('i,i', mass, x)/total_mass
    CoMy = np.einsum('i,i', mass, y)/total_mass
    CoMz = np.einsum('i,i', mass, z)/total_mass
    return np.array([CoMx,CoMy,CoMz])

def xyz_reader(name):
    file = open(name,'r')
    lines = file.readlines()
    lines.pop(0)
    lines.pop(0)
    
    Z = np.array([])
    x = np.array([])
    y = np.array([])
    z = np.array([])

    for element in lines:
        a_line_in_lines = element.split()
        element_number = np.where(Z_dictonary==(element[0] + element[1]))
    
        Z = np.append(Z, element_number[0], axis = 0)
        x = np.append(x, [Bohr(float(a_line_in_lines[1]))], axis = 0)
        y = np.append(y, [Bohr(float(a_line_in_lines[2]))], axis = 0)
        z = np.append(z, [Bohr(float(a_line_in_lines[3]))], axis = 0)
    file.close()
    
    return np.stack((Z, x, y, z))

def output_xyz(Z, x, y, z):
    composition_Z = np.unique(Z, return_counts=True)[0]
    composition_N = np.unique(Z, return_counts=True)[1]
    word = ""
    for index, element in enumerate(composition_Z):
        word = word + " " + Z_dictonary[element] + str(composition_N[index]) 
    word = word + ".xyz"
    word = word.replace(" ", "")

    export_xyz = open(word,"w+")
    export_xyz.write(str(len(Z)) + "\r\n")
    export_xyz.write("\r\n")
    for i in range(len(Z)):
        export_xyz.write(str(Z_dictonary[Z[i]]) + " " + str(np.around(Ångström(x[i]),6)) + " " + str(np.around(Ångström(y[i]),6)) + " " + str(np.around(Ångström(z[i]),6)) + "\r\n")
    export_xyz.close()
    
    return None

def σ_LangmoreSmith(Z, β):
    return 0.0005357*(Z**(1.5))/(β**2)*( 1 - (0.001679*Z)/(β) )

def σ_plasmonCS(λ, β):
    return ((np.sqrt(4*np.pi*n))/(nA*(β**2)*(137**2)))*np.log(λ*(β**2)*(137**2)*(3*np.pi*np.pi*n)**(-1/3)/(np.sqrt(1 - β**2)))

def σ_SDinelastic(E):
    return 1/(number_density(1.444, 36444)*Bohr(1430/(eV(E)**2) + 0.54*np.sqrt(eV(E))))

def nn(t):
    return (8*np.log(2)/2*np.pi)**(0.5)*(Γt)**(-1)*Number_beam_particles*np.exp( - 4*np.log(2)*(t - Γt)**2/ Γt**2)


######################## Initialize Coordinates
Zxyz = xyz_reader('Ne_1000.xyz')

Z = Zxyz[0]
Z = Z.astype(int) ### convert to an integer numpy array
Rx_old = Zxyz[1]
Ry_old = Zxyz[2]
Rz_old = Zxyz[3]

######################## 1-parameters
α = Z_α[Z]
Eion = Z_Eion[Z]
mass = Z_mass[Z] ## mass in a.u.
valance = Z_valance[Z]
a = np.vstack((Z_a[Z], np.ones(Z_a[Z].shape))).T
b = np.vstack((-Z_valance[Z], np.zeros(Z_valance[Z].shape))).T

nA = 4/(np.sqrt(2)*2.54*2*α[0]**(1/7))**3 ### in a.u. only works for homogenous solids
n = valance[0]*nA ### in a.u.
crystal_area = (len(Rx_old)/nA)**(2/3) ### assuming a cubic crystal
number_e_over_sample = 0.5*scipy.special.erf( np.sqrt(8*np.log(2))*(len(Rx_old)/nA)**(1/3)/(2*Γx*np.sqrt(2))) - 0.5*scipy.special.erf( -np.sqrt(8*np.log(2))*(len(Rx_old)/nA)**(1/3)/(2*Γx*np.sqrt(2)))

######################## Inital Distances and Basis Vectors
XX = np.einsum('j, i -> ij', Rx_old, np.ones(Rx_old.shape)) - np.einsum('i, j -> ij', Rx_old, np.ones(Rx_old.shape))
YY = np.einsum('j, i -> ij', Ry_old, np.ones(Ry_old.shape)) - np.einsum('i, j -> ij', Ry_old, np.ones(Ry_old.shape))
ZZ = np.einsum('j, i -> ij', Rz_old, np.ones(Rz_old.shape)) - np.einsum('i, j -> ij', Rz_old, np.ones(Rz_old.shape))
rr = (XX**2 + YY**2 + ZZ**2)**(0.5)

ex = Get_basis_vectors(rr, XX)
ey = Get_basis_vectors(rr, YY)
ez = Get_basis_vectors(rr, ZZ)

######################## Inital Acceleration
radial_force = force(rr, α, Eion)
ax_old = np.einsum('ij, ij, i -> i', radial_force, ex, mass**(-1))
ay_old = np.einsum('ij, ij, i -> i', radial_force, ey, mass**(-1))
az_old = np.einsum('ij, ij, i -> i', radial_force, ez, mass**(-1))

######################## Inital Velocity
speed = sps.maxwell.rvs((Temperature/mass)**(1/2), size=len(Rx_old))/mass
θ = np.random.random_sample(len(Rx_old))*(np.pi)
φ = np.random.random_sample(len(Rx_old))*(2.)*(np.pi)
vx_old = speed*np.sin(θ)*np.cos(φ)
vy_old = speed*np.sin(θ)*np.sin(φ)
vz_old = speed*np.cos(θ)

######################## Graph These
rrr = np.linspace(0, 10, num=int(10/0.1)) ### This is the bin vector used for RDFs (in Ångström)

### Keep Track of These
KE_t = np.array([np.einsum('i, i->', vx_old**2 + vy_old**2 + vz_old**2, mass/2)])
PE_t = np.array([energy(rr, α, Eion)])
RDF_t = np.array([Get_RDF(rr, rrr)])
RMSD_t = np.array([0])
Elastic_t = np.array([0])
Plasmon_t = np.array([0])
Exciton_t = np.array([0])

###
CoM_0 = Get_CoM(mass, Rx_old, Ry_old, Rz_old)
x0 = Rx_old
y0 = Ry_old
z0 = Rz_old

speed_t = np.sqrt(np.einsum('i -> ', vx_old)**2 + np.einsum('i -> ', vy_old)**2 + np.einsum('i -> ', vz_old)**2)

for element in time_array[1:]:
    start = time.time()
    
    ############ MC: Cross Sections
    σ_elastic = σ_LangmoreSmith(Z, β_beam) ### https://srdata.nist.gov/SRD64/Elastic
    σ_plasmon = σ_plasmonCS(λ, β_beam)*np.ones(len(Rx_old))
    σ_exciton = σ_SDinelastic(energy_z) - σ_plasmon
    
    P_elastic = 1 - (1 - σ_elastic/crystal_area)**(nn(element)*number_e_over_sample*number_e_over_sample)
    P_plasmon = 1 - (1 - σ_plasmon/crystal_area)**(nn(element)*number_e_over_sample*number_e_over_sample)
    P_exciton = 1 - (1 - σ_exciton/crystal_area)**(nn(element)*number_e_over_sample*number_e_over_sample)
    
    Dice1 = np.random.rand(len(Rx_old))
    Dice2 = np.random.rand(len(Rx_old))
    Dice3 = np.random.rand(len(Rx_old))
    
    hit_elastic = np.less(Dice1, P_elastic)
    hit_plasmon = np.less(Dice2, P_exciton)
    hit_exciton = np.less(Dice3, P_exciton)
    
    itemindex_elastic = np.where(hit_elastic==True)
    itemindex_plasmon = np.where(hit_plasmon==True)
    itemindex_exciton = np.where(hit_exciton==True)
    
    ############ Modeling Plasmon decay into phonons
    N_plasmons = len(itemindex_plasmon[0]) # 0
    scale = np.sqrt((KE_t[-1] + N_plasmons*np.sqrt(4*np.pi*n))/(KE_t[-1]))
    vx_old = scale*vx_old
    vy_old = scale*vy_old
    vz_old = scale*vz_old
    
    ############ Exciton Excitations
    excitons = itemindex_exciton[0]
    a[excitons, 1] = Z_exciton[Z[excitons]] #0.1902
    b[excitons] = b[excitons] + [1, -1]
    #α[excitons] = [20] ## estimated ion polarizability
    #Eion[excitons] = [0.779] ## new dictonary which requires the 2nd ionization energy
    
    ############ Position Integration
    Rx_new = Rx_old + vx_old*dt + 0.5*dt*dt*ax_old
    Ry_new = Ry_old + vy_old*dt + 0.5*dt*dt*ay_old
    Rz_new = Rz_old + vz_old*dt + 0.5*dt*dt*az_old
    
    XX = np.einsum('j, i -> ij', Rx_new, np.ones(Rx_new.shape)) - np.einsum('i, j -> ij', Rx_new, np.ones(Rx_new.shape))
    YY = np.einsum('j, i -> ij', Ry_new, np.ones(Ry_new.shape)) - np.einsum('i, j -> ij', Ry_new, np.ones(Ry_new.shape))
    ZZ = np.einsum('j, i -> ij', Rz_new, np.ones(Rz_new.shape)) - np.einsum('i, j -> ij', Rz_new, np.ones(Rz_new.shape))
    rr = (XX**2 + YY**2 + ZZ**2)**(0.5)
    
    ex = Get_basis_vectors(rr, XX)
    ey = Get_basis_vectors(rr, YY)
    ez = Get_basis_vectors(rr, ZZ)
    
    ############ Force Calcuation
    radial_force = force(rr, α, Eion) + Coulomb_force(rr, a, b, np.abs(valance))
    np.fill_diagonal(radial_force, 0)
    ax_new = np.einsum('ij, ij, i -> i', radial_force, ex, mass**(-1))
    ay_new = np.einsum('ij, ij, i -> i', radial_force, ey, mass**(-1))
    az_new = np.einsum('ij, ij, i -> i', radial_force, ez, mass**(-1))
    
    ############ Velocity Integration
    vx_new = vx_old + 0.5*dt*(ax_old + ax_new)
    vy_new = vy_old + 0.5*dt*(ay_old + ay_new)
    vz_new = vz_old + 0.5*dt*(az_old + az_new)
    
    ############ Old = New
    Rx_old = Rx_new
    Ry_old = Ry_new
    Rz_old = Rz_new
    ax_old = ax_new
    ay_old = ay_new
    az_old = az_new
    vx_old = vx_new
    vy_old = vy_new
    vz_old = vz_new
    
    ############ Keep Track of These 
    KE_t = np.append(KE_t, [np.einsum('i, i->', vx_new**2 + vy_new**2 + vz_new**2, mass/2)], axis = 0)
    PE_t = np.append(PE_t, [energy(rr, α, Eion) + Coulomb_energy(rr, a, b, np.abs(valance))  ], axis = 0)
    RDF_t = np.append(RDF_t, [Get_RDF(rr, rrr)], axis = 0)
    RMSD_t = np.append(RMSD_t, [Get_RMSD(x0, y0, z0, Rx_old, Ry_old, Rz_old)/(speed_t*element)], axis = 0)
    Elastic_t = np.append(Elastic_t, [len(itemindex_elastic[0])], axis = 0)
    Plasmon_t = np.append(Plasmon_t, [len(itemindex_plasmon[0])], axis = 0)
    Exciton_t = np.append(Exciton_t, [len(itemindex_exciton[0])], axis = 0)

    end = time.time()
    Total_CPU = Total_CPU + end - start 
    print("Time Step " + str(int(element/dt)) + " CPU Time: " + str(end - start) + " s")

outfile = 'output_Ar.npz'
np.savez(outfile, Rx_old, Ry_old, Rz_old, vx_old, vy_old, vz_old, ax_old, ay_old, az_old, mass, Z, α, Eion, a, b)
output_xyz(Z, Rx_new, Ry_new, Rz_new)

plt.plot(fs(time_array), Elastic_t, 'r--', label='Elastic')
plt.plot(fs(time_array), Plasmon_t, 'b--', label='Plasmon')
plt.plot(fs(time_array), Exciton_t, 'g--', label='Exciton')
plt.plot(fs(time_array), 1.5*np.amax(Exciton_t)*nn(time_array)/(np.amax(nn(time_array))), 'r', label='Beam')
plt.legend()
plt.title('Events')
plt.xlabel('Time [fs]')
plt.ylabel('Events')
plt.savefig("Events_t.png", dpi=150)
plt.show()

plt.plot(rrr[1:], RDF_t[0], 'b--', label='initial')
plt.plot(rrr[1:], RDF_t[-1], 'r--', label='final')
plt.legend()
plt.title('RDF')
plt.xlabel('Distance [Å]')
plt.ylabel('RDF [a.u.]')
plt.savefig("RDF_fi.png", dpi=150)
plt.show()

"""
plt.plot(fs(time_array), RMSD_t, 'b--', label='x_t')
plt.legend()
plt.title('RMSD')
plt.xlabel('Time [fs]')
plt.ylabel('RMSD [a.u.]')
plt.savefig("RMSD_t.png", dpi=150)
plt.show()
"""

plt.plot(fs(time_array), KE_t, 'b--', label='Kinetic')
plt.plot(fs(time_array), PE_t, 'g--', label='Potential')
plt.plot(fs(time_array), KE_t+PE_t, 'r--', label='Total')
plt.legend()
plt.title('Energy')
plt.xlabel('Time [fs]')
plt.ylabel('Energy [a.u.]')
plt.savefig("Energy_t.png", dpi=150)
plt.show()

###### Plot 4: The Dynamic RDF: g(r,t)
x = rrr
y = fs(time_array)
X,Y = np.meshgrid(x,y)
Z = RDF_t

plt.pcolormesh(X,Y,Z)
plt.title('g(r,t)')
plt.xlabel('Distance [Å]')
plt.ylabel('Time [fs]')
plt.colorbar()
plt.savefig("RDF_t.png", dpi=150)
plt.show()

print("Number of Excitons Excited: " + str(np.einsum('i->', Exciton_t)))
print("Number of Plasmons Excited: " + str(np.einsum('i->', Plasmon_t)))
print("Dose Received: " + str((KE_t[-1] + PE_t[-1]) - (KE_t[0] + PE_t[0])))