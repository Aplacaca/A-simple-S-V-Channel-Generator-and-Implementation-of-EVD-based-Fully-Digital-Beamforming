'''
Author: dhy
Mail: git config user.email
Date: 2022-11-21 11:09:36
LastEditors: dhy
LastEditTime: 2022-11-22 14:35:39
FilePath: /PyMIMO/gen_array_response.py
Description:

Copyright (c) 2022 by dhy 3057931787@qq.com, All Rights Reserved. 
'''

import numpy as np

def array_square_response(a1,a2,N):
    # firstly we adopt square antenna to calculate array response
    # a1 H_angle
    # a2 V_angle
    y = np.zeros(N,dtype=np.complex64)
    H_antenna_num = np.sqrt(N).astype(np.int64)
    V_antenna_num = np.sqrt(N).astype(np.int64)
    for m in range(0,H_antenna_num):
        for n in range(0,V_antenna_num):
            y[m*(H_antenna_num)+n] = np.exp( 1j* np.pi * (m*np.sin(a1)*np.sin(a2) + n*np.cos(a2)))
    y = y/np.sqrt(N)

    return y

def array_response(H_angle,V_angle,H_antenna_num,V_antenna_num):
    # firstly we adopt square antenna to calculate array response
    # a1 H_angle azimuth step np.sin(H_angle)*np.sin(V_angle)
    # a2 V_angle elevation step np.cos(V_angle)
    # spacing d = λ/2
    # y = np.zeros(H_antenna_num*V_antenna_num,dtype=np.complex128)
    # for m in range(0,V_antenna_num):
    #     for n in range(0,H_antenna_num):
    #         y[m*(H_antenna_num)+n] = np.exp( 1j* np.pi * (n*np.sin(H_angle)*np.sin(V_angle) + m*np.cos(V_angle)))
    # y = y/np.sqrt(H_antenna_num*V_antenna_num)
    vector_V = np.exp(1j * np.arange(start=0, stop=V_antenna_num, step=1, dtype=np.complex128) * np.pi * np.cos(V_angle))
    vector_H = np.exp(1j * np.arange(start=0, stop=H_antenna_num, step=1, dtype=np.complex128) * np.pi *np.sin(H_angle) * np.sin(V_angle))
    beams = np.kron(vector_V,vector_H)
    y = beams/np.sqrt(H_antenna_num*V_antenna_num)
    return y

if __name__=='__main__':
    import pdb
    y,beams = array_response(H_angle=1,V_angle=1,H_antenna_num=2,V_antenna_num=3)
    pdb.set_trace()