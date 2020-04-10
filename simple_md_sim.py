#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:16:57 2020

@author: julio
"""

import numpy as np
import scipy 
import scipy.stats as sps
import matplotlib.pyplot as plt
from jcp_rgs_md import *

###############################################################################

### Enviroment Inputs
Temperature = 50 #/315774.64
T = 1.01e4 ### total duration in fs
dt = 2.5 ### Time Step in fs
Zxyz = xyz_reader('50___Ar4631.xyz').T ### XYZ file for simulation
γ = 0.0001 # Langevin Coupling (in a.u.)

### Beam Inputs
energy_z = 1000000 ### forward beam particle energy in eV
fluence = 10 ### in electrons/Ångström^2
Γx = 5000 ### beam radius in Ångströms

###############################################################################

Z_dictonary = np.array(['e ', 'H ', 'He', 
                        'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',
                        'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])
Z_mass = np.array([ 1.,   1837.,   7297.,  
                   12650., 16427.,  19705.,  21894.,  25533.,  29164.,  34631., 36785.,  
                   41908., 44305.,  49185.,  51195.,  56462.,  58441.,  64621.,  72820.,  
                   71271., 73057.,  81949.,  87256.,  92861.,  94782., 100145., 101799., 107428., 106990., 115837., 119180., 127097., 132396., 136574., 143955., 145656., 152754.,
                   155798., 159721., 162065., 166291., 169357., 174906., 176820., 184239., 187586., 193991., 196631., 204918., 209300., 216395., 221954., 232600., 231331., 239332., 
                   242270., 250331., 253208., 255415., 256859., 262937., 264318., 274089., 277013., 286649., 289702., 296219., 300649., 304894., 307947., 315441., 318945., 325367., 329848., 335119., 339434., 346768., 350390., 355616., 359048., 365656., 372561., 377702., 380947., 380983., 382806., 404681.,
                   406504., 411972., 413795., 422979., 421152., 433900., 432024., 444784., 442961., 450253., 450253., 457545., 459367., 468482., 470305., 472128., 477596., 486711., 492179., 490357., 492179., 492179., 506763., 512231., 512231., 519523., 521346., 526814., 526814., 534106., 534106., 535929.])

####### Initialize Coordinates with an additional particle at infinity (1e50)
Z = np.append([118], Zxyz.T[0].astype(int))
mass = Z_mass[Z]
Rx = Bohr(np.append([1e50], Zxyz.T[1])) # Zxyz.T[1] 
Ry = Bohr(np.append([1e50], Zxyz.T[2])) # Zxyz.T[2]
Rz = Bohr(np.append([1e50], Zxyz.T[3])) # Zxyz.T[3]
onez = np.ones(Z.shape)

####### Initial Complete Pair-wise Distance Matrix
XX = np.einsum('j, i -> ij', Rx, onez) - np.einsum('i, j -> ij', Rx, onez)
YY = np.einsum('j, i -> ij', Ry, onez) - np.einsum('i, j -> ij', Ry, onez)
ZZ = np.einsum('j, i -> ij', Rz, onez) - np.einsum('i, j -> ij', Rz, onez)
rr = (XX**2 + YY**2 + ZZ**2)**(0.5)

####### Initial RDF
rrr = np.linspace(0, 10, num=int(10/0.1)) ### This is the bin vector used for RDFs (in Ångström)
RDF_t = np.array([Get_RDF(rr, rrr)])

####### Neighbor-list Implementation
extract = 80
R_neighbor_indices = np.argsort(rr, axis=0)
r_index = R_neighbor_indices[1:extract,:] #extract the 1st nearest to the exract nearest, ingore the 0th nearest (itself)! #η = r_index.transpose()

reduced_XX = np.repeat([Rx],(extract-1),axis=0) - Rx[r_index] ### this gets the neighbors for every atom
reduced_YY = np.repeat([Ry],(extract-1),axis=0) - Ry[r_index]
reduced_ZZ = np.repeat([Rz],(extract-1),axis=0) - Rz[r_index]
reduced_R = (reduced_XX**2 + reduced_YY**2 + reduced_ZZ**2)**(0.5)

r_index[reduced_R > Bohr(10.0)] = 0

####### Force Calculation
radial_force = force(reduced_R, 10.6, 0.579)
ax_old = np.einsum('ai, ai, i -> i', radial_force, reduced_XX, mass**(-1))
ay_old = np.einsum('ai, ai, i -> i', radial_force, reduced_YY, mass**(-1))
az_old = np.einsum('ai, ai, i -> i', radial_force, reduced_ZZ, mass**(-1))

####### Initial Velocities, KE, and PE.
speed = sps.maxwell.rvs((Temperature/mass)**(0.5), size=len(mass))/mass
θ = np.random.random_sample(len(mass))*(np.pi)
φ = np.random.random_sample(len(mass))*(2.)*(np.pi)
vx = speed*np.sin(θ)*np.cos(φ)
vy = speed*np.sin(θ)*np.sin(φ)
vz = speed*np.cos(θ)
KE_t = np.array([0.5*np.einsum('i, i -> ', mass[1:], vx[1:]**2 + vy[1:]**2 + vz[1:]**2)])
PE_t = np.array([0.5*np.einsum('ni ->', energy(reduced_R[:,1:], 10.6, 0.579))])
####### States
array_of_wannier_excitons = np.array([], dtype=int)
exciton_time_stamp = np.array([])
number_of_plasmons_t = np.array([], dtype=int)

####### PES
Ar_PES = np.load("Ar_PES.npy")
eng_pes = Ar_PES[0]
force_arar_pes =  Ar_PES[1] 

####### Generated Parameters
Temperature = au_temperature(Temperature)
energy_z = Hartree(energy_z)
T = au_time(T)
dt = au_time(dt)
time_array = np.linspace(0., T, num=int(T/dt))

Γx = Bohr(Γx)
Γt = T/2 ## FWHM in au_time
nA = 4/((2**(0.5))*2.54*2*10.6**(1/7))**3 ### in a.u. only works for homogenous solids for fcc = 4
n = 8*nA ### number density in a.u.
crystal_area = (len(Z)/nA)**(2/3) ### assuming a cubic crystal
number_e_over_sample = 0.5*scipy.special.erf( ((8*np.log(2))**0.5)*(len(Z)/nA)**(1/3)/(2*Γx*(2**0.5))) - 0.5*scipy.special.erf( -((8*np.log(2))**(0.5))*(len(Z)/nA)**(1/3)/(2*Γx*(2**0.5)))
spot_size = 2*np.sqrt(Γx*Γx) ### Beam Spot Diameter
beam_area = np.pi*Γx*Γx ### Beam Area
fluence = fluence/3.57106483 ### fluence in electrons/Bohr^2
Number_beam_particles = int(beam_area*fluence)

################## System PES Selection
FF_1types = np.array([0, 1, 2], dtype=int)
FF_2types = np.array([[0,1],[1,2]], dtype=int)

System_1types = np.zeros((len(Z),), dtype=int)
System_2typez = np.stack([np.repeat([System_1types],(extract-1),axis=0), System_1types[r_index]], axis=2) 
System_2types = FF_2types[System_2typez[:,:,0], System_2typez[:,:,1]]

for element in time_array[1:]:
    ### Langevin Thermostat
    ξx = np.random.normal(0, 1, len(Rx)) ## 0 mean and 1 = σ^2 (variance = 1)
    ξy = np.random.normal(0, 1, len(Ry))
    ξz = np.random.normal(0, 1, len(Rz))
    θx = np.random.normal(0, 1, len(Rx))
    θy = np.random.normal(0, 1, len(Ry))
    θz = np.random.normal(0, 1, len(Rz))
    
    Cx_t = 0.5*(dt**2)*(ax_old - γ*vx) + np.sqrt(2*Temperature*γ/mass)*(dt**(1.5))*(0.5*ξx + 0.5*θx/(np.sqrt(3)))
    Cy_t = 0.5*(dt**2)*(ay_old - γ*vy) + np.sqrt(2*Temperature*γ/mass)*(dt**(1.5))*(0.5*ξy + 0.5*θy/(np.sqrt(3)))
    Cz_t = 0.5*(dt**2)*(az_old - γ*vz) + np.sqrt(2*Temperature*γ/mass)*(dt**(1.5))*(0.5*ξz + 0.5*θz/(np.sqrt(3)))
    
    Rx = Rx + vx*dt + Cx_t
    Ry = Ry + vy*dt + Cy_t
    Rz = Rz + vz*dt + Cz_t
    
    ### Exciton Poisson Decay
    Probability_of_Decay = 1 - np.exp(np.log(0.5)*(element - exciton_time_stamp)/(4132231400.))
    hit_decay = np.less(np.random.rand(len(Probability_of_Decay)), Probability_of_Decay)
    itemindex_decay = np.where(hit_decay==True)
    System_1types[array_of_wannier_excitons[itemindex_decay]] = 0
    exciton_time_stamp[itemindex_decay] = element
    
    ### Stochastic Excitations Cross Sections
    P_elastic = 1 - (1 - σ_LangmoreSmith(Z, β_beam(energy_z))/crystal_area)**(NN(element, Γt, Number_beam_particles)*number_e_over_sample*number_e_over_sample)
    P_plasmon = 1 - (1 - 2*σ_LangmoreSmith(Z, β_beam(energy_z))/crystal_area)**(NN(element, Γt, Number_beam_particles)*number_e_over_sample*number_e_over_sample)  #1 - (1 - 1.5*σ_LangmoreSmith(Z, β_beam)/crystal_area)**(nn(element)*number_e_over_sample*number_e_over_sample)
    P_exciton = 1 - (1 - 3*σ_LangmoreSmith(Z, β_beam(energy_z))/crystal_area)**(NN(element, Γt, Number_beam_particles)*number_e_over_sample*number_e_over_sample)  #0.000001 #1 - (1 - 1.5*σ_LangmoreSmith(Z, β_beam)/crystal_area)**(nn(element)*number_e_over_sample*number_e_over_sample)
    
    hit_elastic = np.less(np.random.rand(len(Z)), P_elastic)
    hit_plasmon = np.less(np.random.rand(len(Z)), P_exciton)
    hit_exciton = np.less(np.random.rand(len(Z)), P_exciton)
    
    itemindex_elastic = np.where(hit_elastic==True)
    itemindex_plasmon = np.where(hit_plasmon==True)
    itemindex_exciton = np.where(hit_exciton==True)

    ### Plasmon Excitation
    number_of_plasmons = len(itemindex_plasmon[0])
    number_of_plasmons_t = np.append(number_of_plasmons_t, [number_of_plasmons], axis = 0)
    scale = ((KE_t[-1] + number_of_plasmons*np.sqrt(4*np.pi*n))/(KE_t[-1]))**(0.5)
    vx = scale*vx
    vy = scale*vy
    vz = scale*vz
    
    ### Exciton Excitation
    System_1types[itemindex_exciton] = 1
    array_of_wannier_excitons = np.append(array_of_wannier_excitons, (itemindex_exciton[0]).astype(int), axis = 0)
    exciton_time_stamp = np.append(exciton_time_stamp, element*np.ones(len(itemindex_exciton[0])), axis = 0)
    
    ##### New Distance Calculaton
    reduced_XX = np.repeat([Rx],(extract-1),axis=0) - Rx[r_index]
    reduced_YY = np.repeat([Ry],(extract-1),axis=0) - Ry[r_index]
    reduced_ZZ = np.repeat([Rz],(extract-1),axis=0) - Rz[r_index]
    reduced_R = (reduced_XX**2 + reduced_YY**2 + reduced_ZZ**2)**(0.5)

    ##### Types
    System_2typez = np.stack([np.repeat([System_1types],(extract-1),axis=0), System_1types[r_index]], axis=2) 
    System_2types = FF_2types[System_2typez[:,:,0], System_2typez[:,:,1]]

    #### Force Calculation
    dist_index_temp = 110*np.copy(reduced_R)
    dist_index_temp[reduced_R > 20.] = -1
    distance_index = (dist_index_temp).astype(int)
    radial_force = force_london_morse(reduced_R, 10.6, 0.579) + force_arar_pes[System_2types, distance_index] 

    radial_force[radial_force > 10.] = 0
    ax_new = np.einsum('ai, ai, i -> i', radial_force, reduced_XX, mass**(-1))
    ay_new = np.einsum('ai, ai, i -> i', radial_force, reduced_YY, mass**(-1))
    az_new = np.einsum('ai, ai, i -> i', radial_force, reduced_ZZ, mass**(-1))
    
    #### Velocity Integration
    vx = vx + 0.5*dt*(ax_old + ax_new) - γ*vx*dt + ξx*(np.sqrt(2*Temperature*γ*dt/mass)) + γ*Cx_t
    vy = vy + 0.5*dt*(ay_old + ay_new) - γ*vy*dt + ξy*(np.sqrt(2*Temperature*γ*dt/mass)) + γ*Cy_t
    vz = vz + 0.5*dt*(az_old + az_new) - γ*vz*dt + ξz*(np.sqrt(2*Temperature*γ*dt/mass)) + γ*Cz_t
    
    #### Acceleration Reset
    ax_old = ax_new
    ay_old = ay_new
    az_old = az_new
    
    PE_chag = 0.5*np.einsum('ia ->', eng_pes[System_2types,asd])
    RDF_t = np.append(RDF_t, [Get_RDF(reduced_R, rrr)], axis = 0)
    KE_t = np.append(KE_t, [0.5*np.einsum('i, i -> ', mass[1:], vx[1:]**2 + vy[1:]**2 + vz[1:]**2)], axis = 0)
    PE_t = np.append(PE_t, [0.5*np.einsum('ni -> ', energy(reduced_R[:,1:], 10.6, 0.579)) - PE_chag ], axis = 0)

output_xyz(Z[1:], Rx[1:], Ry[1:], Rz[1:])

TE = KE_t+PE_t
energy_deposited = np.abs(TE[1] - np.max(TE))
total_mass = np.einsum('i-> ', mass[1:]) 
print("Energy Deposited = " + str(np.round( eV( energy_deposited ) , 1)) + " eV")
print("Dose Deposited = " + str( np.round(kGy(energy_deposited/total_mass), 1 )) + " kGy")
print("Fluence = " + str(Bohr(Bohr(fluence))) + " e/Å^2")

###### Plot 1: The trRDF: g(r,t)
x = rrr
y = ps(time_array)
X,Y = np.meshgrid(x,y)
Zed = RDF_t

plt.pcolormesh(X,Y,Zed)
plt.title('trRDF')
plt.xlabel('Distance (Å)')
plt.ylabel('Time (ps)')
plt.colorbar()
plt.savefig("RDF_t_.jpg", dpi=150)
plt.show()

###### Plot 2: Energy
plt.plot(ps(time_array), KE_t, 'b--', label='Kinetic')
plt.plot(ps(time_array), PE_t, 'g-.', label='Potential')
plt.plot(ps(time_array), KE_t+PE_t, 'r-', label='Total')
plt.legend()
plt.title('Energy')
plt.xlabel('Time (ps)')
plt.ylabel('Energy (a.u.)')
plt.savefig("Energy_t_.jpg", dpi=150)
plt.show()