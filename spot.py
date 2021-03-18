# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:53:39 2021

@author: sehoc
"""


import os
os.chdir(r'C:\Users\sehoc\OneDrive\Documents\GitHub\spotify_analysis')

def numeric_only(data):
    numeric_data = data.drop(columns=['artists',
                                      'id',
                                      'name',
                                      'release_date'])
    return numeric_data

def split_data(data):
    numeric_data = numeric_only(data)
    poplist = numeric_data['popularity'].tolist()
    y_data = []
    for i in range(0, len(data)):
        if poplist[i] > 42:
            y_data.append(1)
        else:
            y_data.append(0)
    x_data = numeric_data.drop(columns=['popularity'])
    return x_data, y_data