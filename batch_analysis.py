# -*- coding: utf-8 -*-
"""
============================================
# Created: 12/13/2022
# Edited: 02/16/2023
# Author: Siddhesh Naidu

This code runs the IFEWs model analysis for a chain of inputs that can be read from an excel file

Changelog:
- Updated static values of RCN_c, A_corn, A_soy, AH_corn, AH_soy to use dynamic data values available per year per county from 1997-2019
- Updated the physics equations to neglect the effect of RCN_soy as per new information from Julia's Energy Methodology paper. The old lines have been commented out for now.
- Updated Looping structure to linearly solve for missing data points in RCN_c, A_corn, A_soy, AH_corn, AH_soy
============================================
"""

import pandas as pd
import time
import os

# Import all functions from OpenMDAO model module
from IFEWs_model import *

# Logs the Starting Time for the Code
start_time = time.time()

# Disables Report Generation as the Model is Run Iteratively
os.environ['OPENMDAO_REPORTS'] = 'False'

# Constants
May_P = 80 # May planting progress averaged at 80%  for 2009 - 2019

## Read CSV Files

# Load Weather data -----------
loc_W = 'Data_Input/weather_data/PRISM_199701_201912_sorted.csv' # Weather dataframe
df_W = pd.read_csv(loc_W)

# Load Animal Agriculture data -----------
loc_AA = 'Data_Input/animal_agriculture_data/IFEW_Counties_1997_2019.csv'
df_AA = pd.read_csv(loc_AA) # Animal Agricultural dataframe

# Parse Weather Data
T_July = df_W['tmean (degrees F)'][(df_W['Month']==7)]
PPT_June = df_W['ppt (inches)'][(df_W['Month']==6)]
PPT_July = df_W['ppt (inches)'][(df_W['Month']==7)]

# Parse Agricultural Data
RCN_c = df_AA["CommercialN_kg_ha"]
A_corn = df_AA["CornAcresPlanted"]
A_soy = df_AA["SoybeansAcresPlanted"]
AH_corn = df_AA["CornGrainAcresHarvested"]
AH_soy = df_AA["SoybeansAcresHarvested"]

# Parse Animal Agricultural Data
cattle_B = df_AA["BeefCows"]
cattle_M = df_AA["MilkCows"]
cattle_H = df_AA["Hogs"]
cattle_O = df_AA["OtherCattle"]


for i in range(len(df_AA)):

    # Checks for missing data and replaces it with the average of the previous two entries
    if pd.isna(RCN_c.iloc[i]) == True:
        df_AA.at[i, "CommercialN_kg_ha"] = (RCN_c.iloc[i - 1] + RCN_c.iloc[i - 2]) / 2
    if pd.isna(A_corn.iloc[i]) == True:
        df_AA.at[i, "CornAcresPlanted"] = (A_corn.iloc[i - 1] + A_corn.iloc[i - 2]) / 2
    if pd.isna(A_soy.iloc[i]) == True:
        df_AA.at[i, "SoybeansAcresPlanted"] = (A_soy.iloc[i - 1] + A_soy.iloc[i - 2]) / 2
    if pd.isna(AH_corn.iloc[i]) == True:
        df_AA.at[i, "CornGrainAcresHarvested"] = (AH_corn.iloc[i - 1] + AH_corn.iloc[i - 2]) / 2
    if pd.isna(AH_soy.iloc[i]) == True:
        df_AA.at[i, "SoybeansAcresHarvested"] = (AH_soy.iloc[i - 1] + AH_soy.iloc[i - 2]) / 2

    x = [RCN_c.iloc[i], A_corn.iloc[i], A_soy.iloc[i], AH_corn.iloc[i], AH_soy.iloc[i], cattle_H.iloc[i], cattle_B.iloc[i], cattle_M.iloc[i], cattle_O.iloc[i]] 
    w = [May_P, T_July.iloc[i], PPT_July.iloc[i], PPT_July.iloc[i] ** 2, PPT_June.iloc[i]]

    raw_results = IFEW(x, w, False)
    
    if i == 0:
        ns_data = [raw_results[0]]
        yc_data = [raw_results[1]]
        ys_data = [raw_results[2]]
        Et_data = [raw_results[3]]
    else:
        ns_data =  ns_data + [raw_results[0]]
        yc_data =  yc_data + [raw_results[1]]
        ys_data =  ys_data + [raw_results[2]]
        Et_data =  Et_data + [raw_results[3]]

# Creates Directory
if not os.path.exists("Data_Output"):
    os.mkdir("Data_Output")

# Saves outputs as text file
df_result = pd.DataFrame({"N2 Surplus" : ns_data, "Corn Yield" : yc_data, "Soybean Yield" : ys_data, "Ethanol Production" : Et_data})
df_result.to_csv("Data_Output/Results.csv", index=False)

# # Additional Intermediate Outputs
# df_result1 = pd.DataFrame({"RCN_c" : RCN_c, "A_corn" : A_corn, "A_soy" : A_soy, "AH_corn" : AH_corn, "AH_soy" : AH_soy})
# df_result1.to_csv("Data_Output/InterimOutputs.csv", index=False)

# Logs the Ending Time for the Code
end_time = time.time()

# Prints the Execution Time
exec_time = end_time - start_time
print("\nExecution Time =", str(exec_time), "\n")
# %%