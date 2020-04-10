#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 16:02:38 2020

@author: julio
"""
import numpy as np
import scipy

def Get_RDF(R, rr): ### δr and r_cutoff in Ångström
    #binz = np.linspace(0, r_cutoff, num=int(r_cutoff/δr)) ### in Ångström
    binz = Bohr(rr) ### in Bohr
    shell_vol = np.diag(np.repeat([4*np.pi*binz**3/3],len(binz),axis=0) - np.transpose(np.repeat([4*np.pi*binz**3/3],len(binz),axis=0)), k=1) ### shell_volume = 4*np.pi*binz**3/3
    his = np.histogram(R.flatten(), bins=binz) ### Make R-Matrix into 1D array, we should get len(R) number of 0s and then duplicates of the rest!
    RDF = his[0]/shell_vol/his[0][0]
    RDF[0] = 0
    #RDF = np.insert(RDF, 0, 0., axis=0)
    return RDF # 175 s.t. the first peak is at 14, the neighborhood for FCC

def Coulomb_energy(r, a, b, Z):
    one = np.ones(r.shape)
    np.fill_diagonal(one, 0)
    
    aa = np.sqrt(np.einsum('ia, jb -> ijab', a, a)/(np.einsum('ia, jb -> ijab', a, np.ones(a.shape)) + np.einsum('ia, jb -> ijab', np.ones(a.shape), a)))
    
    E_nn = np.einsum('i, j, ij, ij -> ', Z, Z, (r + np.identity(len(Z)))**(-1), one)/2
    E_ee = np.einsum('ia, jb, ijab, ij, ij -> ', b, b, scipy.special.erf(np.einsum('ijab, ij -> ijab', aa, r)), (r + np.identity(len(Z)))**(-1), one)/2
    E_ne = np.einsum('j, ib, ijb, ij, ij -> ', Z, b, scipy.special.erf(np.einsum('ib, ij -> ijb', a, r)), (r + np.identity(len(Z)))**(-1), one)/2
    E_en = np.einsum('i, ja, ija, ij, ij -> ', Z, b, scipy.special.erf(np.einsum('ja, ij -> ija', a, r)), (r + np.identity(len(Z)))**(-1), one)/2
    
    return E_nn + E_ee + E_ne + E_en

def xyz_reader(name): #currentlogfile = 'Ar_1000.xyz'#'Ne_1000.xyz'#'Na_7.xyz'#'Na_1000.xyz'# #currentlogfile = 'Xe_1Million.xyz' #currentlogfile = 'ArTest20.xyz' #ArTest.xyz
    Z_dictonary = np.array(['e ', 'H ', 'He', 
                        'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',
                        'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])

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
        x = np.append(x, [(float(a_line_in_lines[1]))], axis = 0)
        y = np.append(y, [(float(a_line_in_lines[2]))], axis = 0)
        z = np.append(z, [(float(a_line_in_lines[3]))], axis = 0)
    file.close()
    
    return np.stack((Z, x, y, z))

def Bohr(x): ### Given Ångström get Bohr
    return 1.88973*x
def Ångström(x): ### Given Bohr get Ångström
    return x/1.88973 #
def fs(x): ### Given time in a.u. get fs
    return x/41.341447 #
def Hartree(x): ### in eV get Hartree
    return x/27.2114
def eV(x): ### in Hartree get eV
    return x*27.2114
def kGy(x): ### given dose in a.u. (# Dose (a.u.) 4.78599307e12 = Dose (S.I.))
    return x*4.78599307e12/(1e3)
def au_temperature(x): ### given Temperature in Kelvin
    return x/315774.64
def au_time(x): ### given fs
    return x*41.34137 
def ps(x): ### given a.u. time
    return x*0.00002418884

def β_beam(E):
    return np.sqrt(1 - 1/((1 + E*(1/137)**2)**2))

def force(r, α, E):
    Rmin = 2.54*2*α**(1/7)
    C6 = 3/2*α*α*E*E/(2*E)
    MB = np.exp(-(6/Rmin)*(r-Rmin)) #return (C6/(Rmin)**7)*(12*np.exp(-(12/Rmin)*(r-Rmin)) - 2*6*np.exp(-(6/Rmin)*(r-Rmin))) #/Rmin
    return 12*(C6/(Rmin)**7)*(MB**2 - MB)

def force_Morse(r, k, dissociation_energy, Rmin):
    MB = np.exp(-(np.sqrt(k/(2*dissociation_energy)))*(r-Rmin)) #return (C6/(Rmin)**7)*(12*np.exp(-(12/Rmin)*(r-Rmin)) - 2*6*np.exp(-(6/Rmin)*(r-Rmin))) #/Rmin
    return np.sqrt(2*k*dissociation_energy)*(MB**2 - MB)

def force_FF(r, System_k1, System_k2, System_k3):
    MB = np.exp(-(0.5*System_k2/System_k1)**(0.5)*(r - System_k3))
    return np.sqrt(2*System_k2*System_k1)*(MB**2 - MB)

def energy_FF(r, System_k1, System_k2, System_k3):
    MB = np.exp(-(0.5*System_k2[:,1:]/System_k1[:,1:])**(0.5)*(r[:,1:] - System_k3[:,1:]))
    return System_k1[:,1:]*(MB**2 - 2*MB)

def energy(r, α, E):
    Rmin = 2.54*2*α**(1/7)
    C6 = 3/2*α*α*E*E/(2*E)
    MB = np.exp(-(6/Rmin)*(r-Rmin))
    return (C6/(Rmin)**6)*(MB**2 - 2*MB)

def Get_RDF(R, rr): ### δr and r_cutoff in Ångström
    #binz = np.linspace(0, r_cutoff, num=int(r_cutoff/δr)) ### in Ångström
    binz = Bohr(rr) ### in Bohr
    shell_vol = np.diag(np.repeat([4*np.pi*binz**3/3],len(binz),axis=0) - np.transpose(np.repeat([4*np.pi*binz**3/3],len(binz),axis=0)), k=1) ### shell_volume = 4*np.pi*binz**3/3
    his = np.histogram(R.flatten(), bins=binz) ### Make R-Matrix into 1D array, we should get len(R) number of 0s and then duplicates of the rest!
    RDF = his[0]/shell_vol/his[0][0]
    RDF[0] = 0
    #RDF = np.insert(RDF, 0, 0., axis=0)
    return RDF # 175 s.t. the first peak is at 14, the neighborhood for FCC

def output_xyz(Z, x, y, z):
    Z_dictonary = np.array(['e ', 'H ', 'He', 
                        'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 
                        'Na', 'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar',
                        'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
                        'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 
                        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U ', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'])    
    
    composition_Z = np.unique(Z, return_counts=True)[0]
    composition_N = np.unique(Z, return_counts=True)[1]
    word = "___"
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

def force_london_morse(r, α, E):
    Rmin = 2.54*2*α**(1/7)
    C6 = 3/2*α*α*E*E/(2*E)
    MB = np.exp(-(6/Rmin)*(r-Rmin)) #return (C6/(Rmin)**7)*(12*np.exp(-(12/Rmin)*(r-Rmin)) - 2*6*np.exp(-(6/Rmin)*(r-Rmin))) #/Rmin
    return 12*(C6/(Rmin)**7)*(MB**2 - MB)

def force_london_morse___(r, α, E):
    Rmin = 2.54*2*α**(1/7)
    C6 = 3/2*α*α*E*E/(2*E)
    MB = np.exp(-(6/Rmin)*(r-Rmin)) #return (C6/(Rmin)**7)*(12*np.exp(-(12/Rmin)*(r-Rmin)) - 2*6*np.exp(-(6/Rmin)*(r-Rmin))) #/Rmin
    return 12*(C6/(Rmin)**7)*(MB**2 - MB)

def NN(t, Γt, Number_beam_particles):
    return (8*np.log(2)/2*np.pi)**(0.5)*(Γt)**(-1)*Number_beam_particles*np.exp( - 4*np.log(2)*(t - Γt)**2/ Γt**2)

def σ_LangmoreSmith(Z, β):
    return 0.0005357*(Z**(1.5))/(β**2)*( 1 - (0.001679*Z)/(β) )
