'''
Author: dhy 3057931787@qq.com
Date: 2022-11-21 09:03:31
LastEditors: dhy
LastEditTime: 2022-11-22 15:18:55
FilePath: /PyMIMO/gen_channel.py
Description: 

Copyright (c) 2022 by dhy 3057931787@qq.com, All Rights Reserved. 
'''

import numpy as np
from gen_array_response import array_response
import random

random.seed(777)

Ns = 3 # of streams
Nc = 5 # of clusters
Nray = 10 # of rays in each cluster
Nt_H = 2
Nt_V = 8
Nt = Nt_H*Nt_V # of transmit antennas
Nr_H = 2
Nr_V = 2
Nr = Nr_H*Nr_V # of receive antennas

N_sub_path = int(Nc*Nray)

angle_sigma = 10/180*np.pi # standard deviation of the angles in azimuth and elevation both of Rx and Tx
gamma = np.sqrt((Nt*Nr)/(Nc*Nray)) # normalization factor
sigma = 1 # according to the normalization condition of the H

realization = 1000 # Time?
count = 0

AoA = np.zeros((2, Nc*Nray))
AoD = np.zeros((2, Nc*Nray))

# initialize arrays
H = np.zeros((Nr,Nt,realization),dtype=np.complex64)
At = np.zeros((Nt,Nc*Nray,realization),dtype=np.complex64)
Ar = np.zeros((Nr,Nc*Nray,realization),dtype=np.complex64)
alpha = np.zeros((Nc*Nray,realization),dtype=np.complex64)
Fopt =  np.zeros((Nt,Ns,realization),dtype=np.complex64)
Wopt =  np.zeros((Nr,Ns,realization),dtype=np.complex64)

for reali in range(realization):
    # generate AoA/AoD data for each cluster
    for c in range(Nc):
        AoD_m = np.random.uniform(0,2*np.pi,size=(1,2))
        AoA_m = np.random.uniform(0,2*np.pi,size=(1,2))
        # Note: should divide angle_sigma by sqrt 2 to adopt numpy.random.laplace api
        # AoA/AoD is from a laplace distribution of mean m and std angle_sigma
        AoD[0,c*Nray:Nray*(c+1)] = AoD_m[:,0] + np.random.laplace(loc=AoD_m[:,0],scale=angle_sigma/np.sqrt(2),size=(1,Nray)) 
        AoD[1,c*Nray:Nray*(c+1)] = AoD_m[:,1] + np.random.laplace(loc=AoD_m[:,1],scale=angle_sigma/np.sqrt(2),size=(1,Nray)) 
        AoA[0,c*Nray:Nray*(c+1)] = AoA_m[:,0] + np.random.laplace(loc=AoA_m[:,0],scale=angle_sigma/np.sqrt(2),size=(1,Nray)) 
        AoA[1,c*Nray:Nray*(c+1)] = AoA_m[:,1] + np.random.laplace(loc=AoA_m[:,1],scale=angle_sigma/np.sqrt(2),size=(1,Nray)) 
    
    # calculate H
    for j in range(0,N_sub_path):
        At[:,j,reali] = array_response(AoD[0,j],AoD[1,j],Nt_H,Nt_V) # UPA array response
        Ar[:,j,reali] = array_response(AoA[0,j],AoA[1,j],Nr_H,Nr_V)
        alpha[j,reali] = np.random.normal(0,np.sqrt(sigma/2)) + 1j*np.random.normal(0,np.sqrt(sigma/2))
        H[:,:,reali] = H[:,:,reali] + alpha[j,reali] * np.expand_dims(Ar[:,j,reali],axis=-1) @ np.expand_dims(np.conjugate(At[:,j,reali].T),axis=0)

    H[:,:,reali] = gamma * H[:,:,reali]
    
    if(np.linalg.matrix_rank(H[:,:,reali])>=Ns):
        count = count + 1
        [U,S,V] = np.linalg.svd(H[:,:,reali],full_matrices=True)
        Fopt[:,:,reali] = V[0:Nt,0:Ns]
        Wopt[:,:,reali] = U[0:Nr,0:Ns]
