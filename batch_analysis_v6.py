#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:05:53 2024

# Author: Siddhesh Naidu and Taeho Jeong
# Updated by: Taeho Jeong 

This code runs the IFEWs model analysis for a chain of inputs that can be read from an excel file
This code uses IFEWs_model_v5

============================================
"""

import numpy as np
import pandas as pd
import time
import os

# from NN_county_yield_fertilizer import * 
# from Cvg_County import *
# from imp_function import *
from IFEWs_model_v6_1 import IFEW

# Logs the Starting Time for the Code
start_time = time.time()

# Disables Report Generation as the Model is Run Iteratively
os.environ['OPENMDAO_REPORTS'] = 'False'

# Constants
May_P = 80 # May planting progress averaged at 80%  for 2009 - 2019
RCN_c = 185  # Sawyer(2018) Avg (155-215)
RCN_s = 17 # Avg = 17.7 kg/ha std = 4.8kg/ha based on the fertilizer use and price data between 2008-2018 (USDA, 2019)

# Read CSV Files

# Load Weather data -----------
loc_W = 'weather_data/PRISM_199701_201912_sorted.csv' # Weather dataframe
df_W = pd.read_csv(loc_W)

# Load Animal Agriculture data -----------
loc_AA = 'animal_agriculture_data/IFEW_Counties_1968_2019.csv'
df_AA = pd.read_csv(loc_AA) # Animal Agricultural dataframe
filtered_data = df_AA[(df_AA['Year'] >= 1997) & (df_AA['Year'] <= 2019)]

# Parse Weather Data
T_July = df_W['tmean (degrees F)'][(df_W['Month']==7)]
PPT_June = df_W['ppt (inches)'][(df_W['Month']==6)]
PPT_July = df_W['ppt (inches)'][(df_W['Month']==7)]


# # From old data 1997-2019
# cattle_B = df_AA["BeefCows"]
# cattle_M = df_AA["MilkCows"]                
# cattle_H = df_AA["Hogs"]
# cattle_O = df_AA["OtherCattle"]

# A_soy = df_AA["SoybeansAcresPlanted"]
# AH_soy = df_AA["SoybeansAcresHarvested"] 
# A_corn = df_AA["CornAcresPlanted"]
# AH_corn = df_AA["CornGrainAcresHarvested"]

# From new data 1968-2019
# From new data 1968-2019
cattle_B = filtered_data["beef"]
cattle_M = filtered_data["milk"]                
cattle_H = filtered_data["hogs"]
cattle_O = filtered_data["cattle"]

A_soy = filtered_data["soy_pa"]
AH_soy = filtered_data["soy_ha"] 
A_corn = filtered_data["corng_pa"]
AH_corn = filtered_data["corng_ha"]


for i in range(len(filtered_data)):
    x = [RCN_c, RCN_s, cattle_H.iloc[i], cattle_B.iloc[i], cattle_M.iloc[i], cattle_O.iloc[i]] 
    w = [May_P, T_July.iloc[i], PPT_July.iloc[i], PPT_July.iloc[i] ** 2, PPT_June.iloc[i]]
    e = [A_corn.iloc[i], AH_corn.iloc[i], A_soy.iloc[i], AH_soy.iloc[i]]
    
    raw_results = IFEW(x, w, e, False)
    
    # if i == 0:
    #     ns_data = [raw_results[0]]
    #     yc_data = [raw_results[1]]
    #     ys_data = [raw_results[2]]
    #     CN = [raw_results[3]]
    #     MN = [raw_results[4]]
    #     FN = [raw_results[5]]
    #     GN = [raw_results[6]]
    #     P_EtOH_data = [raw_results[7]]
    # else:
    #     ns_data =  ns_data + [raw_results[0]]
    #     yc_data =  yc_data + [raw_results[1]]
    #     ys_data =  ys_data + [raw_results[2]]
    #     CN = CN + [raw_results[3]]
    #     MN = MN + [raw_results[4]]
    #     FN = FN + [raw_results[5]]
    #     GN = GN + [raw_results[6]]
    #     P_EtOH_data = P_EtOH_data + [raw_results[7]]

    if i == 0:
        ns_data = [raw_results[0]]
        yc_data = [raw_results[1]]
        ys_data = [raw_results[2]]
        # RCN_c = [raw_results[3]]
        # RCN_s = [raw_results[4]]
        Hog = [raw_results[5]]
        Catt = [raw_results[6]]
        A_c = [raw_results[7]]
        A_s = [raw_results[8]]
        AH_c = [raw_results[9]]
        AH_s = [raw_results[10]]
        CN = [raw_results[11]]
        MN = [raw_results[12]]
        FN = [raw_results[13]]
        GN = [raw_results[14]]
    else:
        ns_data =  ns_data + [raw_results[0]]
        yc_data =  yc_data + [raw_results[1]]
        ys_data =  ys_data + [raw_results[2]]
        # RCN_c = RCN_c  + [raw_results[3]]
        # RCN_s = RCN_s  + [raw_results[4]]
        Hog = Hog + [raw_results[5]]
        Catt = Catt + [raw_results[6]]
        A_c = A_c + [raw_results[7]]
        A_s = A_s + [raw_results[8]]
        AH_c = AH_c + [raw_results[9]]
        AH_s = AH_s + [raw_results[10]]
        CN = CN + [raw_results[11]]
        MN = MN + [raw_results[12]]
        FN = FN + [raw_results[13]]
        GN = GN + [raw_results[14]]

# df_result = pd.DataFrame({"N Surplus" : ns_data, "Corn Yield" : yc_data, "Soy Yield" : ys_data, 
#                           "Commercial N (CN)" : CN, "Manure N (MN)" : MN, "Fixed N (FN)" : FN, 
#                           "Grain N (GN)" : GN, "Ethanol Production" : P_EtOH_data })
# df_result = pd.DataFrame({"N Surplus" : ns_data, "Commercial N (CN)" : CN, "Manure N (MN)" : MN, 
#                           "Fixed N (FN)" : FN, "Grain N (GN)" : GN})
df_result = pd.DataFrame({"N Surplus" : ns_data,"y_c" : yc_data, "y_s" : ys_data, "RCN_c" : RCN_c, "RCN_s" : RCN_s, "Hog" : Hog, 
                          "Cattle" : Catt, "A_c" : A_c, "A_s" : A_s, "AH_c" : AH_c, "AH_s" : AH_s,
                          "Commercial N (CN)" : CN, "Manure N (MN)" : MN, 
                          "Fixed N (FN)" : FN, "Grain N (GN)" : GN})

df_result.to_csv("Results_new.csv", index=False)


# Logs the Ending Time for the Code
end_time = time.time()

# Prints the Execution Time
exec_time = end_time - start_time
print("\nExecution Time =", str(exec_time), "\n")



'''
Post-process data
'''

df = pd.read_csv('Results_new.csv')

df_cleaned = df.dropna(subset=['N Surplus'])

df_cleaned = df_cleaned[df_cleaned['Cattle'] > 0]
df_cleaned = df_cleaned[df_cleaned['Hog'] > 0]

df_cleaned.to_csv('Results_new.csv', index=False)
