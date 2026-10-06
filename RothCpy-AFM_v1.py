# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:20:23 2025
@author: Francesco Palazzi

###############################################################################
                            RothCpy-AFM
A Python version of the RothC model accounting for different C-input sources
in agricultural and agroforestry systems across diverse pedoclimatic conditions
###############################################################################

#  MODIFIED VERSION FROM THE ORIGINAL PYTHON CODE BY COLEMAN
#  & ADAPTATION OF THE MODEL DEVELOPED BY Roberta Farina
#
#  STATIC input data
#  sand:     sand content of the soil       (%)
#  silt:     silt content of the soil       (%)
#  clay:     clay content of the soil       (%)
#  skeleton: skeleton content of the soil   (%)
#  depth: depth of topsoil                  (cm)
#  Corg_pct: organic C of the soil          (%) USE IF SOC (MgC/ha) IS UNKNOWN 
#  Corg_ton: organic C of the soil          (MgC/ha) in the 30 cm layer; if UNKNOWN LEAEV BLANK
#  IOMknown: FLAG condition         (IF UNKNOWN, the code computes it from previous soil properties)
#  IOM: inert organic matter                (MgC/ha)
#  nsteps: number of timesteps 
#
#  DYNAMIC input data
#  year:    year
#  month:   month (1-12)
#  TMP:     (°C) Air temperature
#  Rain:    (mm) Rainfall                                                      must be a CUMULATED VALUE
#  Irrig:   (mm) Irrigation, if in one month there is NO irrigation, specify   (0)
#  Evap:    (mm) Open-pan evaporation OR potential evapotranspiration (mm)          must be a CUMULATED VALUE
#  C_inp:   carbon input to the soil each month (units: t C /ha)
#  FYM:     Farmyard manure input to the soil each month (units: t C /ha)
#  PC:      Plant cover (0 = no cover, 1 = covered by a crop)
#
#  OUTPUTS - All pools are carbon and not organic matter (IOM)
#  DPM:   Decomposable Plant Material (units: t C /ha)
#  RPM:   Resistant Plant Material    (units: t C /ha)
#  BIO:   Microbial Biomass           (units: t C /ha)
#  HUM:   Humified Organic Matter     (units: t C /ha)
#  IOM:   Inert Organic Matter        (units: t C /ha)
#  TOC:   Soil Organic Matter / Total organic Matter (units: t C / ha)
#
#  Other parameters computed/considered:
#  SWC:         soil moisture deficit (mm per soil depth)
#  RM_TMP:      rate modifying fator for temperature (0.0 - ~5.0)
#  RM_Moist:    rate modifying fator for moisture (0.0 - 1.0)
#  RM_PC:       rate modifying fator for plant retainment (0.6 or 1.0)
===============================================================================
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from tqdm import tqdm  


# TODO - select input path folder
input_folder  = r"C:\user\path\Input_FOLDER_optional"

# TODO - select output path folder; create it if doesn't exist
output_folder  = r"C:\user\path\Output_FOLDER_optional"
os.makedirs(output_folder, exist_ok=True)

# TODO IMPORTANT!!! specify type of model 
print('\n\nFirst you must specify the Type of model you want to use to simulate SOC dynamics\n\n')
print('Type in rothc_def to use the standard RothC model by Coleman\n')
print('Type in rothc_10n to use the modified version of RothC\n')
model = input() 
print('Model selected: ', model, '\n')

#%% FUNCTIONS that will be used throughout the code

# FUNCTION to approximate decimal values
def approx_decimals (val1, 
                     val2, 
                     precision):
    return all(round(a, precision) == round(b, precision) for a, b in zip(val1, val2))
#endfunction


# FUNCTION provinding information over soil
def get_soil_class(sand: float, 
                   silt: float, 
                   clay: float) -> str:
    # check that the sum of soil texture values are = 100
    txtr_sum = sand + silt + clay
    txtr_sum = round(txtr_sum, 2)
    if txtr_sum == 100:
        print("\nTexture values check: passed (sum of values = 100)")
    else:
        sys.exit("\nCheck texture values, the sum of sand + silt + clay is NOT 100")
    #endif
    
    # start of conditions
    if clay > 60:
        return "Very Fine"
    elif clay > 35:
        return "Fine"
    elif clay > 18:
        if sand < 15:
            return "Medium Fine"
        elif sand < 65:
            return "Medium"
        else:
            return "Medium"
        #endif
    else:
        if sand < 15:
            return "Medium Fine"
        elif sand > 65:
            return "Coarse"
        else:
            return "Medium"
        #endif
    #endif
#endfunction



# FUNCTION to get IOM (remove duplicates of code if the commands are elsewhere in this script)
def get_IOM(df_head):
    sand        = df_head.loc[0, "sand"] #sand can also be neglected from data, or kept in storage if other PTFs are adopted in future
    silt        = df_head.loc[0, "silt"]
    clay        = df_head.loc[0, "clay"]
    
    # check to make sure sum of txtr values = 100
    txtr_sum = sand + silt + clay
    txtr_sum = round(txtr_sum, 2)
    if txtr_sum == 100:
        print("\nTexture values check: passed (sum of values = 100)")
    else:
        sys.exit("\nCheck texture values, the sum of sand + silt + clay is NOT 100")
    #endif
    
    if pd.isna(df_head.loc[0, "skeleton"]):
        skel = 0
        print("\nSkeleton value not fetched: assegnato default value = 0")
    else:
        skel = df_head.loc[0, "skeleton"]
    
    depth       = df_head.loc[0, "depth"]
    
    Corg        = df_head.loc[0, "Corg_pct"]
    Corg_tC     = df_head.loc[0, "Corg_ton"]
    IOMknown    = df_head.loc[0, "IOMknown"]
    
    # From soil properties compute hydraulic props
    # First convert C-org (in %) Soil Organic Matter using van bemmelen factor
    SOM = Corg * 1.724
    # Bulk Density derived using Rawls & Brakensiek method
    BD = 1.51 + 0.0025*(100-silt-clay) - 0.0013*(100-silt-clay)*SOM - 0.0006*clay*SOM - 0.0048*(clay**2)/60
    
    # Initial SOC stock (Corg_init)
    '''IF IOM is known, just type it into the input file and the code will use it,
    otherwise, you can provide initial SOC ...
    OR let the model compute it from the soil properties
    '''
    if IOMknown == 0 and math.isnan(Corg):                                     # unknown IOM + unknown CORG in %
        # initial Corg (i.e. SOC stock in MgC/ha) is PROVIDED by the user
        Corg_init = Corg_tC
        # IOM computation
        IOMinit = 0.049 * np.power(Corg_init, 1.139)
        print("Initial SOC: %.6f MgC/ha\n"%Corg_init)
        print("IOM content: %.6f\n"%IOMinit)
    elif IOMknown == 0 and math.isnan(Corg_tC):                                # unknown IOM + unknown SOC in MgC/ha
        # initial Corg (i.e. SOC stock in MgC/ha) must be DERIVED from soil properties
        Corg_init = (BD*(Corg/100)*(30/100)*(100-skel)/100)*10000
        # IOM computation
        IOMinit = 0.049 * np.power(Corg_init, 1.139)
        print("Initial SOC: %.6f MgC/ha\n"%Corg_init)
        print("IOM content: %.6f\n"%IOMinit)
    else:                                                                      # known IOM, Corg_init derived via reverse formula
        # IOM 
        IOMinit = df_head.loc[0, "IOM"]#.astype(float)
        Corg_init = np.power(IOMinit / 0.049, 1.0 / 1.139) #inverse formula to calculate Corg_init
        print("Initial SOC: %.6f MgC/ha\n"%Corg_init)
        print("IOM content: %.6f\n"%IOMinit)
    #endif
    return IOMinit, Corg_init
#endfunction


# FUNCTION to compute RATE MODIFYING FACTORS (RMFs)
def RMFs(df, 
         df_head, 
         model): # df_head replaces clay, depth, ETknown
    # the first part of function is not related to the model adopted
    sand        = df_head.loc[0, "sand"]
    silt        = df_head.loc[0, "silt"]
    clay        = df_head.loc[0, "clay"]
    depth       = df_head.loc[0, "depth"]
    Corg        = df_head.loc[0, "Corg_pct"]
    Corg_tC     = df_head.loc[0, "Corg_ton"]
    IOMknown    = df_head.loc[0, "IOMknown"]
    ETknown     = df_head.loc[0, "ETknown"]
    
    # double check that the sum of soil texture values are = 100
    txtr_sum = sand + silt + clay
    txtr_sum = round(txtr_sum, 2)
    if txtr_sum == 100:
        print("\nTexture values check: passed (sum of values = 100)\n")
    else:
        sys.exit("\nCheck texture values, the sum of sand + silt + clay is NOT 100\n")
    #endif
    
    # additional check on soil texture
    soil_class = get_soil_class(sand, silt, clay)
    print("\nSoil class is: ", soil_class)
    
    # From soil properties compute hydraulic props
    # First convert C-org (in %) Soil Organic Matter
    SOM = Corg * 1.724
    # Bulk Density derived using Rawls & Brakensiek method
    BD = 1.51 + 0.0025*(100-silt-clay) - 0.0013*(100-silt-clay)*SOM - 0.0006*clay*SOM - 0.0048*(clay**2)/60
    
    # RATE MODIFYING FACTORS - TEMPERATURE
    # Conditions to define the daily RMF for t°                                Remains EQUAL between RothC-def and RothC-10N
    df.loc[df['Temp'] < -5.0, 'RMF_t'] = 0.0
    df.loc[df['Temp'] > -5.0, 'RMF_t'] = 47.91/(1.0 + np.exp(106.06/(df['Temp']+18.27)))
    
    # RATE MODIFYING FACTORS - PLANT COVER
    # Conditions to define the daily RMF for plant cover (Y/N=1/0)             Remains EQUAL between RothC-def and RothC-10N
    df.loc[df['PC'] == 0, 'RMF_pc'] = 1.0
    df.loc[df['PC'] == 1, 'RMF_pc'] = 0.6
    
    # RATE MODIFYING FACTORS - SOIL MOISTURE                                   Here things get a bit different
    # First compute SOIL MOISTURE DEFICIT (SMD)                                This part of SMD remains EQUAL between RothC-def and RothC-10N 
    if ETknown == 0: #value is still provided as EVAPORATION, must be converted into ET
        df['SMD'] = df['Water'] - 0.75 * df['Evap']
    elif ETknown == 1: # value is already expressed as ET
        df['SMD'] = df['Water'] - df['Evap']
    #endif
    
    # RMF thresholds for soil moisture                                         Different between RothC-def and RothC-10N
    if model == 'rothc_def':
        # RMF thresholds-------------------------------------------------------COLEMAN VERSION OF MODEL
        RMFmax = 1.0
        RMFmin = 0.2 
        
        # Soil moisture deficit computations - Coleman
        SMDMax = -(20 + (1.3 * clay) - (0.01 * (clay**2)))
        SMDMaxAdj = (depth / 23.0) * SMDMax
        SMDBare = SMDMaxAdj / 1.8
        SMD1bar = SMDMaxAdj * 0.444
        
        # Then the monthly cumulated SMD is calcualted
        df.loc[0, "SMDacc"] = 0     #initial condition, modified immediately by lines below
        for i in range(1, len(df)):
            # rule to assign value from previous day
            prev = i-1
            # get the cumulated SMD (even to eventually elaborate graphs in the output section)
            if df['PC'].loc[i] == 1:    #soil is vegetated
                df.loc[i, 'SMDacc'] = max(SMDMaxAdj, 
                                          min(0.0, df.loc[prev, 'SMDacc'] + df.loc[i, 'SMD']))
            else: #PC = 0               #bare soil
                df.loc[i, 'SMDacc'] = max(min(SMDBare, df.loc[prev, 'SMDacc']),
                                          min (0.0, df.loc[prev, 'SMDacc'] + df.loc[i, 'SMD']))
            #endif
        #endfor
                
        #NEW:
        # default condition (SMD1bar > SMDMaxAdj)
        df['RMF_smd'] = 1.0
        # condition to assign in case values are between SMD1bar & SMDMaxAdj
        mask = (df['SMDacc'] < SMD1bar) & (df['SMDacc'] >= SMDMaxAdj)
        df.loc[mask, 'RMF_smd'] = RMFmin + (RMFmax - RMFmin) * (SMDMaxAdj - df.loc[mask, 'SMDacc']) / (SMDMaxAdj - SMD1bar)
        # condition in case values < SMDMaxAdj
        df.loc[df['SMDacc'] < SMDMaxAdj, 'RMF_smd'] = RMFmin
        #----------------------------------------------------------------------COLEMAN VERSION OF MODEL ends here
    
    #BUT if condition continues
    elif model == 'rothc_10n':
        # RMF thresholds-------------------------------------------------------RothC-10N VERSION OF MODEL
        RMFmax = 1.0
        RMFmin = 0.1
        
        ''' Soil moisture deficit computations - 10N (considers VanGenuchten hydraulic formulas)
        In VanGenuchten first are compute the thetas and other coefficients,
        then the params to compute SMD & RMF values to use in RothC are derived'''
        
        # But first, if texture values for sand and silt are not provided, the script must stop
        for col in ["sand", "silt"]:
            if col not in df_head.columns or df_head.empty or pd.isna(df_head.loc[0,col]):
                sys.exit(f"INPUT COMPILATION ERROR: the {col} value is missing")
            #endif
        #endfor
        
        
        # coefficient used in VanGenuchten formulas based on top/subsoil
        if depth == 30:
            t_horizon = 1
        elif depth > 30:
            t_horizon = 0
        #endif
        
        # theta_r       - SoilWaterContent at high tension
        th_r    = 0.01
        # theta_sat     - SoilWaterContent at saturation
        th_sat = (0.7919 
                  + 0.001691*clay - 0.29619*BD 
                  - 0.000001491*(silt**2) + 0.0000821*(SOM**2) 
                  + 0.02427/clay + 0.01113/silt 
                  + 0.01472*np.log(silt ) 
                  - 0.0000733*SOM*clay 
                  - 0.000619*BD*clay 
                  - 0.001183*BD*SOM 
                  - 0.0001664*t_horizon*silt)
        # alpha         - prores distribution
        alpha   = np.exp(
            - 14.96 
            + 0.03135*clay + 0.0351*silt + 0.646*SOM + 15.29*BD - 0.192*t_horizon 
            - 4.671*(BD**2) - 0.000781*(clay**2) - 0.00687*(SOM**2) + 0.0449/SOM 
            + 0.0663*np.log(silt) + 0.1482*np.log(SOM) 
            - 0.04546*BD*silt 
            - 0.4852*BD*SOM 
            + 0.00673*t_horizon*clay)    
        # n             -index of pores' distribution
        n = 1 + np.exp(
            - 25.23 
            - 0.02195*clay + 0.0074*silt - 0.194*SOM + 45.5*BD 
            - 7.24*(BD**2) + 0.0003658*(clay**2) + 0.002885*(SOM**2) 
            - 12.81/BD - 0.1524/silt - 0.01958/SOM 
            - 0.2876*np.log(silt) - 0.0709*np.log(SOM) - 44.6*np.log(BD) 
            - 0.02264*BD*clay 
            + 0.0896*BD*SOM 
            + 0.00718*t_horizon*clay)
        # m             - form parameter
        m = (1-1/n)
        
        # More VanGenuchten-derived params
        FC  = th_sat * (1 + (alpha * 330)**n)**-m
        WP  = th_sat * (1 + (alpha * 15000)**n)**-m
        AWC = FC - WP # that's an additional information, doesn't enter in computations
        
        # NOTE: 1bar=100kPa
        FC_50kPa      = th_r + (th_sat - th_r) * (1/(1+(alpha*50)**n))**m      #50 kPa      = 0.5 bars
        Mb_100KPa     = th_r + (th_sat - th_r) * (1/(1+(alpha*100)**n))**m     #100 kPa     = 1 bars
        M_15000kPa    = th_r + (th_sat - th_r) * (1/(1+(alpha*15000)**n))**m	   #15000 kPa   = 15 bars
        Mc		         = th_r + (th_sat - th_r) * (1/(1+(alpha*1000000)**n))**m #1000000 kPa = 10000 bars
        
        # last effort - SMD coefs/thresholds derived from VanGenuchten on end
        SMD1bar     = (Mb_100KPa - FC_50kPa) * depth * 10     # SMD at 1 bar
        SMDMax      = (M_15000kPa - FC_50kPa) * depth * 10    # SMD at 15 bar
        SMDBare     = SMDMax / 1.8
        SMDMaxAdj   = (Mc - FC_50kPa) * depth * 10            # SMD absolute at 10000 bars (=Mc)
        
        # SMD was already calculated before the IF-cycle
        # now the threshold values change in RothC10N for the cumulated SMD
        df.loc[0, "SMDacc"] = 0     #initial condition, modified immediately by lines below
        for i in range(1, len(df)):
            # rule to assign value from previous day
            prev = i-1
            # get the cumulated SMD (even to eventually elaborate graphs in the output section)
            if df['PC'].loc[i] == 1:     #soil is vegetated
                df.loc[i, 'SMDacc'] = max(SMDMaxAdj, 
                                          min(0.0, df.loc[prev, 'SMDacc'] + df.loc[i, 'SMD']))
            else: #PC = 0               #bare soil
                df.loc[i, 'SMDacc'] = max(min(SMDBare, df.loc[prev, 'SMDacc']), 
                                          min(0.0, df.loc[prev, 'SMDacc'] + df.loc[i, 'SMD']))
            #endif
        #endfor
        
        # Then RMF for SMD can be derived at last
        # Let's start from the extreme condition
        df['RMF_smd'] = RMFmin
        # default condition: SMDacc > SMD1bar (wet ground) -> RMF = 1.0
        mask_wet = df['SMDacc'] > SMD1bar
        df.loc[mask_wet, 'RMF_smd'] = 1.0
        # SMD1bar >= SMDacc > SMDMax (ground starts to dry out)
        mask_dry = (df['SMDacc'] <= SMD1bar) & (df['SMDacc'] > SMDMax)
        df.loc[mask_dry, 'RMF_smd'] = RMFmin + (RMFmax - RMFmin) * (
            (SMDMax - df.loc[mask_dry, 'SMDacc']) / (SMDMax - SMD1bar)
            )
        #----------------------------------------------------------------------RothC-10N VERSION OF MODEL ends here
    #endif
    
    # compute the TOTAL rmf
    df.loc[:, 'RMF_tot'] = df.RMF_t * df.RMF_pc * df.RMF_smd
    
    return df
#endfunction


# FUNCTION to compute C for the different pools
'''This function is the CORE of RothC, once RMFs are computed 
according to the different formulas adopted by the models (RothC default or Rothc-10N)
the computation procedures are the same'''
def C_pools(df, 
            df_head, 
            init_vals):
    clay        = df_head.loc[0, "clay"]        
    #========== step 1 - define coefficients
    
    # DPM/PRM ratio - herbaceous crop
    DPM_RPM_crop = 1.44
    # proportion between DPM & RPM from the DPM/PRM ratio - herbaceous crop ;  DPM_RPM_crop = 1.44
    DPM_crop =  DPM_RPM_crop / (DPM_RPM_crop+1.0)
    RPM_crop =           1.0 / (DPM_RPM_crop+1.0)

    # DPM/PRM ratio - tree crop
    DPM_RPM_tree = 0.30
    # proportion between DPM & RPM from the DPM/PRM ratio - tree crop       ;  DPM_RPM_tree = 0.30
    DPM_tree =  DPM_RPM_tree / (DPM_RPM_tree+1.0)
    RPM_tree =           1.0 / (DPM_RPM_tree+1.0)
        
    
    # proportion of DPM & RPM + HUM for farmyard manure (those are fixed values)
    DPM_fym = 0.49
    RPM_fym = 0.49
    HUM_fym = 0.02
    
    
    # Constant values of decomposition rates for the different pools (DPM/RPM/BIO/HUM)
    # pool = pool(d) * exp(-RMF)
    k_DPM = 10.0
    k_RPM = 0.3
    k_BIO = 0.66
    k_HUM = 0.02
    
    
    # CO2/(BIO+HUM) ratio proportions (function of CLAY content)
    x = 1.67 * (1.85+1.60*np.exp(-0.0786*clay))
    # CO2/(BIO+HUM) ratio proportions (function of CLAY content)            ;  x = 1.67 * (1.85+1.60*np.exp(-0.0786*clay))
    x_bio = (0.46 / (x+1))
    x_hum = (0.54 / (x+1))
    x_co2 = (x / (x+1))
    
    tstep = 12
    
    # sum of x_bio + x_hum + x_co2 MUST BE 1, otherwise simulation has to stop.
    sum_proportions = x_bio + x_hum + x_co2 
    
    # this condition is to avoid errors due to the rounding of decimals
    if not np.isclose(sum_proportions, 1.0, atol=1e-12):
        sys.exit('ERROR: \nThe sum of values of the proportion for each pool must be equal 1')
    #endif
    
    
    #========== COMPUTATION of DPM/RPM/BIO/HUM/CO2
    
    # create new DF to store data of computations importing only necessary columns from original DF
    df_decomp = df[['year', 'month', 'Herb_in', 'Tree_in', 'FYM', 'RMF_tot']].copy()
        
    # insert columns for different parametres that will be computed
    ''' explaination for the different column names
    *_cin = value of DPR/PRM/other from monthly carbon input values (for both herbaceous and tree crops)
    *_fym = value of DPR/PRM/other from monthly farmyard manure values
    *_adj = ADJUSTED value for every pool (PMR/PRM/BIO/etc.)
    *_tot = FINAL value for every pool (PMR/PRM/BIO/etc.)
    *_bio = fraction entering the corresponding pool
    *_hum = fraction entering the corresponding pool
    *_co2 = fraction entering the corresponding pool'''
    df_decomp[['DPM_cin', 'DPM_fym', 'DPM_adj', 'DPM_tot', 'DPM_bio', 'DPM_hum', 'DPM_co2']] = np.nan
    df_decomp[['RPM_cin', 'RPM_fym', 'RPM_adj', 'RPM_tot', 'RPM_bio', 'RPM_hum', 'RPM_co2']] = np.nan
    df_decomp[[                      'BIO_adj', 'BIO_tot', 'BIO_bio', 'BIO_hum', 'BIO_co2']] = np.nan
    df_decomp[[           'HUM_fym', 'HUM_adj', 'HUM_tot', 'HUM_bio', 'HUM_hum', 'HUM_co2']] = np.nan
    df_decomp[[                      'CO2_adj', 'CO2_tot']] = np.nan
    df_decomp[['IOM']] = init_vals[4] #it's the only constant thing in life... apart from death
    
    # Starter row to compute initial values
    start_row = pd.DataFrame(columns=df_decomp.columns)
    # assign NaN to the first row
    start_row.loc[0] = np.nan
    # Then specify proper values 
    start_row.loc[0, 'DPM_tot'] = init_vals[0]
    start_row.loc[0, 'RPM_tot'] = init_vals[1]
    start_row.loc[0, 'BIO_tot'] = init_vals[2]
    start_row.loc[0, 'HUM_tot'] = init_vals[3]
    start_row.loc[0, 'IOM']     = init_vals[4]
    start_row.loc[0, 'CO2_tot'] = init_vals[5]
    
    # add the initial start_row to df_decomp
    df_decomp = pd.concat([start_row, df_decomp], ignore_index=True)
    
    # plant inputs (tree+crop)
    df_decomp.loc[:, 'DPM_cin'] = (df_decomp['Herb_in'] * DPM_crop) + (df_decomp['Tree_in'] * DPM_tree)
    df_decomp.loc[:, 'RPM_cin'] = (df_decomp['Herb_in'] * RPM_crop) + (df_decomp['Tree_in'] * RPM_tree)
    # famryard manure
    df_decomp.loc[:, 'DPM_fym'] = (df_decomp['FYM'] * DPM_fym)
    df_decomp.loc[:, 'RPM_fym'] = (df_decomp['FYM'] * RPM_fym)
    df_decomp.loc[:, 'HUM_fym'] = (df_decomp['FYM'] * HUM_fym)
    
    
    # formula @t0(starter) for ADJ values
    df_decomp.loc[0, 'DPM_adj'] = df_decomp.loc[0, 'DPM_tot'] * np.exp(-df_decomp.loc[1, 'RMF_tot'] * k_DPM / tstep)
    df_decomp.loc[0, 'RPM_adj'] = df_decomp.loc[0, 'RPM_tot'] * np.exp(-df_decomp.loc[1, 'RMF_tot'] * k_RPM / tstep)
    df_decomp.loc[0, 'BIO_adj'] = df_decomp.loc[0, 'BIO_tot'] * np.exp(-df_decomp.loc[1, 'RMF_tot'] * k_BIO / tstep)
    df_decomp.loc[0, 'HUM_adj'] = df_decomp.loc[0, 'HUM_tot'] * np.exp(-df_decomp.loc[1, 'RMF_tot'] * k_HUM / tstep)
    
    # Values that contribute to the computation for the other pools during decomposition
    df_decomp.loc[0, 'DPM_bio'] = (df_decomp.loc[0, 'DPM_tot'] - df_decomp.loc[0, 'DPM_adj']) * x_bio
    df_decomp.loc[0, 'DPM_hum'] = (df_decomp.loc[0, 'DPM_tot'] - df_decomp.loc[0, 'DPM_adj']) * x_hum
    df_decomp.loc[0, 'DPM_co2'] = (df_decomp.loc[0, 'DPM_tot'] - df_decomp.loc[0, 'DPM_adj']) * x_co2
    
    df_decomp.loc[0, 'RPM_bio'] = (df_decomp.loc[0, 'RPM_tot'] - df_decomp.loc[0, 'RPM_adj']) * x_bio
    df_decomp.loc[0, 'RPM_hum'] = (df_decomp.loc[0, 'RPM_tot'] - df_decomp.loc[0, 'RPM_adj']) * x_hum
    df_decomp.loc[0, 'RPM_co2'] = (df_decomp.loc[0, 'RPM_tot'] - df_decomp.loc[0, 'RPM_adj']) * x_co2
    
    df_decomp.loc[0, 'BIO_bio'] = (df_decomp.loc[0, 'BIO_tot'] - df_decomp.loc[0, 'BIO_adj']) * x_bio
    df_decomp.loc[0, 'BIO_hum'] = (df_decomp.loc[0, 'BIO_tot'] - df_decomp.loc[0, 'BIO_adj']) * x_hum
    df_decomp.loc[0, 'BIO_co2'] = (df_decomp.loc[0, 'BIO_tot'] - df_decomp.loc[0, 'BIO_adj']) * x_co2
    
    df_decomp.loc[0, 'HUM_bio'] = (df_decomp.loc[0, 'HUM_tot'] - df_decomp.loc[0, 'HUM_adj']) * x_bio
    df_decomp.loc[0, 'HUM_hum'] = (df_decomp.loc[0, 'HUM_tot'] - df_decomp.loc[0, 'HUM_adj']) * x_hum
    df_decomp.loc[0, 'HUM_co2'] = (df_decomp.loc[0, 'HUM_tot'] - df_decomp.loc[0, 'HUM_adj']) * x_co2
    
    # initial values for CO2 as well
    df_decomp.loc[0, 'CO2_adj'] = (df_decomp.loc[0, 'DPM_co2'] + df_decomp.loc[0, 'RPM_co2'] + 
                                    df_decomp.loc[0, 'BIO_co2'] + df_decomp.loc[0, 'HUM_co2'])
    
    
    # loop to compute values for DPM, RPM, BIO, HUM
    for i in range(1,len(df_decomp)):
        # rules to assign the correct values for RMF
        if i < len(df_decomp) - 1:
            rmf = df_decomp.loc[i+1, 'RMF_tot'] #prehaps switch command with "rmf = df_decomp.loc[i, 'RMF_tot']"
        else:
            # # OLD STRING #rmf = 0  # oppure np.nan o un valore medio, se preferisci
            rmf = df_decomp.loc[i, 'RMF_tot'] # 2026.09.03 debugging
        #endif
        
        # DPM
        df_decomp.loc[i, 'DPM_tot'] = df_decomp.loc[i-1, 'DPM_adj'] + df_decomp.loc[i, 'DPM_cin'] + df_decomp.loc[i, 'DPM_fym']
        df_decomp.loc[i, 'DPM_adj'] = df_decomp.loc[i, 'DPM_tot'] * np.exp(-rmf * k_DPM / tstep)
        df_decomp.loc[i, 'DPM_bio'] = (df_decomp.loc[i, 'DPM_tot'] - df_decomp.loc[i, 'DPM_adj']) * x_bio
        df_decomp.loc[i, 'DPM_hum'] = (df_decomp.loc[i, 'DPM_tot'] - df_decomp.loc[i, 'DPM_adj']) * x_hum
        df_decomp.loc[i, 'DPM_co2'] = (df_decomp.loc[i, 'DPM_tot'] - df_decomp.loc[i, 'DPM_adj']) * x_co2
        
        # RPM
        df_decomp.loc[i, 'RPM_tot'] = df_decomp.loc[i-1, 'RPM_adj'] + df_decomp.loc[i, 'RPM_cin'] + df_decomp.loc[i, 'RPM_fym']
        df_decomp.loc[i, 'RPM_adj'] = df_decomp.loc[i, 'RPM_tot'] * np.exp(-rmf * k_RPM / tstep)
        df_decomp.loc[i, 'RPM_bio'] = (df_decomp.loc[i, 'RPM_tot'] - df_decomp.loc[i, 'RPM_adj']) * x_bio
        df_decomp.loc[i, 'RPM_hum'] = (df_decomp.loc[i, 'RPM_tot'] - df_decomp.loc[i, 'RPM_adj']) * x_hum
        df_decomp.loc[i, 'RPM_co2'] = (df_decomp.loc[i, 'RPM_tot'] - df_decomp.loc[i, 'RPM_adj']) * x_co2
        
        # BIO
        df_decomp.loc[i, 'BIO_tot'] = (df_decomp.loc[i-1, 'BIO_adj'] + 
                                        df_decomp.loc[i-1, 'BIO_bio'] + 
                                        df_decomp.loc[i-1, 'DPM_bio'] + 
                                        df_decomp.loc[i-1, 'RPM_bio'] + 
                                        df_decomp.loc[i-1, 'HUM_bio'])
        df_decomp.loc[i, 'BIO_adj'] = df_decomp.loc[i, 'BIO_tot'] * np.exp(-rmf * k_BIO / tstep)
        df_decomp.loc[i, 'BIO_bio'] = (df_decomp.loc[i, 'BIO_tot'] - df_decomp.loc[i, 'BIO_adj']) * x_bio
        df_decomp.loc[i, 'BIO_hum'] = (df_decomp.loc[i, 'BIO_tot'] - df_decomp.loc[i, 'BIO_adj']) * x_hum
        df_decomp.loc[i, 'BIO_co2'] = (df_decomp.loc[i, 'BIO_tot'] - df_decomp.loc[i, 'BIO_adj']) * x_co2
        
        # HUM
        df_decomp.loc[i, 'HUM_tot'] = (df_decomp.loc[i, 'HUM_fym'] + 
                                        df_decomp.loc[i-1, 'HUM_adj'] + 
                                        df_decomp.loc[i-1, 'DPM_hum'] + 
                                        df_decomp.loc[i-1, 'RPM_hum'] + 
                                        df_decomp.loc[i-1, 'BIO_hum'] + 
                                        df_decomp.loc[i-1, 'HUM_hum'])
        
        df_decomp.loc[i, 'HUM_adj'] = df_decomp.loc[i, 'HUM_tot'] * np.exp(-rmf * k_HUM / tstep)
        df_decomp.loc[i, 'HUM_bio'] = (df_decomp.loc[i, 'HUM_tot'] - df_decomp.loc[i, 'HUM_adj']) * x_bio
        df_decomp.loc[i, 'HUM_hum'] = (df_decomp.loc[i, 'HUM_tot'] - df_decomp.loc[i, 'HUM_adj']) * x_hum
        df_decomp.loc[i, 'HUM_co2'] = (df_decomp.loc[i, 'HUM_tot'] - df_decomp.loc[i, 'HUM_adj']) * x_co2
        
        # CO2
        df_decomp.loc[i, 'CO2_tot'] = df_decomp.loc[i-1, 'CO2_adj'] + df_decomp.loc[i-1, 'CO2_tot']
        df_decomp.loc[i, 'CO2_adj'] = (df_decomp.loc[i, 'DPM_co2'] + 
                                        df_decomp.loc[i, 'RPM_co2'] + 
                                        df_decomp.loc[i, 'BIO_co2'] + 
                                        df_decomp.loc[i, 'HUM_co2'])
        #endif
    #endfor
    
    # Final total SOC (sum of the monthly values for every pool)
    df_decomp.loc[:, 'TOC'] = (df_decomp.loc[:, 'DPM_tot'] + 
                                df_decomp.loc[:, 'RPM_tot'] + 
                                df_decomp.loc[:, 'BIO_tot'] + 
                                df_decomp.loc[:, 'HUM_tot'] + 
                                df_decomp.loc[:, 'IOM'])
    
    last_row = df_decomp.iloc[-1]
    endsim_vals = last_row[['DPM_tot', 'RPM_tot', 'BIO_tot', 'HUM_tot', 'IOM', 'CO2_tot', 'TOC']]
    #convert into list
    new_vals = endsim_vals.tolist()
    return df_decomp, new_vals
#endfunction


# FUNCTIONS for automatic assignmant of Cinputs during SPINUP PHASE - F#1
def run_single_spinup(df_eq_test, 
                      df_head, 
                      init_vals_base):
    '''Launches a spin-up iteration until reaching equilibrium, returning final values
    simulated for SOC and its pools.'''
    init_vals = init_vals_base.copy()
    
    while True:
        init_vals[5] = 0 # reset CO2
        df_decomp, new_vals = C_pools(df_eq_test, df_head, init_vals)
        
        # Condition to stop iteration (equilibrium is reached)
        if approx_decimals(init_vals[:5], new_vals[:5], precision=2):
            break
        #endif
        init_vals = new_vals.copy()
    #endwhile
    toc_final = new_vals[6] # final SOC value at equilibrium
    return toc_final, new_vals
#endfunction

# FUNCTIONS for automatic assignmant of Cinputs during SPINUP PHASE - MAIN
def optimize_c_inputs_for_spinup(df_eq_template, 
                                 df_head, 
                                 Corg_target, 
                                 IOMinit,
                                 tol = None,        # precision of the estimation for the C-inputs
                                 low_c = None,      # lower C-input threshold, usually a good value is around 1
                                 high_c = None,     # upper C-input threshold, usually a good value is around 6
                                 max_iter = None,
                                 herb_pct = None,   # % of C-input assigned to herbaceous residues
                                 tree_pct = None,   # % of C-input assigned to tree residues
                                 sheet_name = ""):
    
    '''By using Bisection, the annual C-input is calculated and optimized 
    to reduce differences from the initial SOC value provided/derived 
    by using the initial soil property values.
    Tolerance is set to 0.01 MgC/ha for the bisection method to assess C-input for spinup'''
    
    # Identify empty cells (NaN) for the 2 C-input columns
    nan_mask_herb = df_eq_template['Herb_in'].isna()
    nan_mask_tree = df_eq_template['Tree_in'].isna()
    
    num_nan_herb = nan_mask_herb.sum()
    num_nan_tree = nan_mask_tree.sum()
    
    
    # Assign values for the PARTITION between tree and herbaceous C-inputs
    # CASE 1: all values filled for 1/2 columns
    if num_nan_herb > 0 and num_nan_tree == 0:
        herb_pct, tree_pct = 1.0, 0.0
    elif num_nan_tree > 0 and num_nan_herb == 0:
        herb_pct, tree_pct = 0.0, 1.0
    else:
        # CASE 2: both or none columns have NaN values 
        h_val = float(herb_pct) if pd.notna(herb_pct) else None
        t_val = float(tree_pct) if pd.notna(tree_pct) else None
        
        if h_val is None and t_val is None:
            # Header totalmente vuoto -> Fallback 50/50
            herb_pct, tree_pct = 0.5, 0.5
        elif h_val is not None and t_val is None:
            herb_pct = min(max(h_val, 0.0), 1.0)
            tree_pct = 1.0 - herb_pct
        elif t_val is not None and h_val is None:
            tree_pct = min(max(t_val, 0.0), 1.0)
            herb_pct = 1.0 - tree_pct
        else:
            # Both columns are filled by the user -> Normalization just in case (es. se inseriti 80 e 20)
            total = h_val + t_val
            if total == 0:
                herb_pct, tree_pct = 0.5, 0.5
            else:
                herb_pct, tree_pct = h_val / total, t_val / total
            #endif
        #endif
    #endif
    
    # If empty cells are absent, only Herb_in is used to adjust C-inputs for spin-up
    fill_all_herb = False
    if num_nan_herb == 0 and num_nan_tree == 0:
        fill_all_herb = True
        num_nan_herb = len(df_eq_template)
    #endif
    
    
    '''Initialise BISECTION calculations    
    bisection is an algorithm that works to find the roots of a function, the
    concept is repeatedly halving an interval until error is reduced to the minimum'''
    
    # initialisation values for spin-up
    init_vals_base = [0, 0, 0, 0, IOMinit, 0]

    # Progress bar for optimization
    pbar = tqdm(total=max_iter,
                desc=f"  ├─ Opt. C-input [{sheet_name}]",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} iter [{elapsed}]",
                )
    best_df_eq = df_eq_template.copy()
    
    # initial fallback value, updated every iteration of the FOR cycle
    optimal_c = 0.0
    
    # BISECTION optimization cycle
    '''For this function it is important to assign:
        - the correct values for C_input target thresholds (low_c & high_c), 
        - the tolerance/accuracy ('tol', i.e., the acceptabel difference between
                                  the SOC value provided as input and 
                                  the value obtained at the end of the spinup)
        - even the number of maximum iterations can be used to sharpen 
        the accuracy of C-inputs estimation'''
    for iteration in range(max_iter):
        # progress bar update
        pbar.update(1)
        
        # BISECTION RULE avg = (val1 + val2) / 2
        mid_c = (low_c + high_c) / 2.0
        
        # dataset to use
        df_test = df_eq_template.copy()

        if fill_all_herb:
            val_per_cell = (mid_c * 10) / num_nan_herb
            df_test["Herb_in"] = val_per_cell
            df_test["Tree_in"] = df_test["Tree_in"].fillna(0.0)
        else:
            if num_nan_herb > 0 and num_nan_tree > 0:
                val_herb = (mid_c * 10 * herb_pct) / num_nan_herb
                val_tree = (mid_c * 10 * tree_pct) / num_nan_tree
                df_test.loc[nan_mask_herb, "Herb_in"] = val_herb
                df_test.loc[nan_mask_tree, "Tree_in"] = val_tree
            elif num_nan_herb > 0:
                val_herb = (mid_c * 10) / num_nan_herb
                df_test.loc[nan_mask_herb, "Herb_in"] = val_herb
                df_test["Tree_in"] = df_test["Tree_in"].fillna(0.0)
            elif num_nan_tree > 0:
                val_tree = (mid_c * 10) / num_nan_tree
                df_test.loc[nan_mask_tree, "Tree_in"] = val_tree
                df_test["Herb_in"] = df_test["Herb_in"].fillna(0.0)
            #endif
        #endif
        df_test["Herb_in"] = df_test["Herb_in"].fillna(0.0)
        df_test["Tree_in"] = df_test["Tree_in"].fillna(0.0)

        toc_eq, _ = run_single_spinup(df_test, df_head, init_vals_base)
        diff = toc_eq - Corg_target

        best_df_eq = df_test
        #update of optimal_c variable
        optimal_c = mid_c

        if abs(diff) <= tol:
            break
        #endif

        if diff < 0:
            low_c = mid_c
        else:
            high_c = mid_c
        #endif
    #endfor
    pbar.close()
    return best_df_eq, optimal_c
#endfunction



#%% Calling data from input file

# CLOCK start global
start_global_time = time.time()

# list of files within directory
files = os.listdir(input_folder)

# filter by the name and .DAT format
input_files = [f for f in files if f.startswith("RothC-AF_") and f.lower().endswith(".xlsx")]


for file_idx, filename in enumerate(input_files, 1):
    input_path = os.path.join(input_folder, filename)
    excel_file = pd.ExcelFile(input_path)
    sheets = excel_file.sheet_names

    print("\n\n" + "="*74)
    print(f"FILE [{file_idx}/{len(input_files)}]: {filename} ({len(sheets)} simulations)")
    print("="*74 + "\n\n")

    # Barra di avanzamento per i Fogli/Campi del file corrente
    sheets_pbar = tqdm(sheets, 
                       desc=f"Progress status: ({file_idx}) field", 
                       unit="field", 
                       leave=True)

    for sheet in sheets_pbar:
        print(f"\n>>> Now processing ")
        sheets_pbar.set_postfix({"Field": sheet})
        
        # CLOCK START processing current sheet
        t_sheet_start = time.time()
        
        #======================================================================
        # PUNCTUAL INPUT DATA (soil + conditions based on the type of input data)
        # Header
        df_head = pd.read_excel(input_path,
                                sheet_name=sheet,
                                skiprows=3, header=0, nrows=1, 
                                index_col=None)
        
        # First derive IOM from input data
        IOMinit, Corg_init = get_IOM(df_head)
        sand        = df_head.loc[0, "sand"]
        silt        = df_head.loc[0, "silt"]
        clay        = df_head.loc[0, "clay"]
        
        
        #======================================================================
        # MONTHLY INPUT DATA
        # Load data from Excel
        df_all = pd.read_excel(input_path,
                               sheet_name=sheet,
                               skiprows = 6)
        
        # Sum of Precipitaiton and Irrigation columns
        loc_index = df_all.columns.get_loc('Irrigation') + 1
        df_all.insert(loc_index, 'Water', df_all['Prec'] + df_all['Irrigation'])
        
        
        #======================================================================
        # Then split it into 2 DFs
        
        
        # FIRST DATASET - used to reach equilibrium
        # the dummy year is repeated 10 times to create a dummy dataset for spinup
        df_eq = pd.concat([df_all[:12].copy()] * 10, ignore_index=True)
        #07/05/2026: pd.concat([df[:12].copy()] * 10, ignore_index=True)        REPLACED w/ pd.concat([df[:12].copy()] * 100 ...
        # compute RMFs for df_eq
        model_spinup = 'rothc_def'
        df_eq = RMFs(df_eq, df_head, model_spinup)
        
        #info on spin-up data for simulation
        print("Number of records of dummy dataset (to reach equilibrium): %.0f\n"%len(df_eq))
        print("Yearly C-input from HERBACEOUS crops until equilibrium: %.2f MgC/ha"%float(df_eq['Herb_in'].sum()/10))
        print("Yearly C-input from TREE crops until equilibrium: %.2f MgC/ha"%float(df_eq['Tree_in'].sum()/10))
        
        
        
        #======================================================================
        # SECOND DATASET - actual years to simulate
        df_sim = pd.concat([df_all[12:].copy()], ignore_index=True)
        print("Number of records (except the years to reach equilibrium): %.0f\n"%len(df_sim))
        
        
        #%%              1ST STEP: SPIN-UP SIMULATION loops until equilibrium is reached
        
        # Step1.1 - C-input optimization---------------------------------------
        # retrieve from input file the proportion between tree and herbaceous inputs 
        # to use for spin-up assignment of C-input
        herb_fraction = df_head.loc[0, "herb_fraction"]
        tree_fraction = df_head.loc[0, "tree_fraction"]
        
        # CLOCK start C-input optimization time
        t_c_opt_start = time.time()        
        
        # COMPUTE C-inputs pre-spinup phase
        df_eq, optimal_annual_c = optimize_c_inputs_for_spinup(df_eq_template=df_eq, 
                                                               df_head=df_head, 
                                                               Corg_target=Corg_init, 
                                                               IOMinit=IOMinit, 
                                                               tol      = 0.5, #!!! evenutally increase/decrease value
                                                               low_c    = 0.5, #!!! adjust accordingly to your case study if necessary
                                                               high_c   = 8.0, #!!! adjust accordingly to your case study if necessary
                                                               max_iter = 10,  #!!! if low_c & high_c range is small, iterations could be reduced
                                                               herb_pct = herb_fraction,
                                                               tree_pct = tree_fraction,
                                                               sheet_name=str(sheet))
        
        # CLOCK end C-input optimization time
        t_c_opt_end = time.time()
        elapsed_c_opt = t_c_opt_end - t_c_opt_start
        
        
        
        # Step1.2 - Actual spinup----------------------------------------------
        # values that must be fetched @t0 to inizialize simulation
        # when the function C_pools was defined initially the variable name was "values"
        
        # CLOCK start actual spinup
        t_spinup_start = time.time()
        
        # in order: DPM, RPM, BIO, HUM,     IOM,  CO2
        init_vals = [0,    0,   0,   0, IOMinit,    0]
        
        # header of .TXT output file where data are stored at the end of every loop iteration until equilibrium
        filename_id = sheet
        output_path = f'{output_folder}/RothC-AF_{filename_id}_out_init_vals_until_eq.txt'
                
        # also print initialising values into the console
        print("\nDPM_tot        RPM_tot       BIO_tot       HUM_tot       IOM     CO2     TOC")
        print(["{:.8f}".format(num) for num in init_vals])
        
        # list where to store the values at the end of every iteration
        history = [init_vals.copy()]
                
        # print("Number of records of dummy dataset (to reach equilibrium): %.0f"%len(df_eq))
        # print("Yearly C-input from HERBACEOUS crops until equilibrium: %.2f MgC/ha"%(df_eq['Herb_in'].sum()/10))
        # print("Yearly C-input from TREE crops until equilibrium: %.2f MgC/ha\n"%(df_eq['Tree_in'].sum()/10))
        
        # on with the loop
        while True:
            # set CO2 value to 0 
            init_vals[5] = 0
            
            # perform simulation
            df_decomp, new_vals = C_pools(df_eq, df_head, init_vals)
            #show on the console the values for the last row (end of year)
            print(["{:.8f}".format(num) for num in new_vals])
            # store a copy of the values of the end of simulation in the "history" list
            history.append(new_vals.copy())
            
            # Values that must reach equilibrium are DPM, RPM, BIO, HUM, IOM, but NOT CO2
            
            # Condition to stop simulation
            
            #       OLD CONDITION
            #     #[:5 considers all columns of df except CO2]
            #     #precision: how many decimal points consider while approximating (excel considers 15 decimal points)
            #     #07/05/2026: precision set to 8 (15 is useless with float64 data)
            if approx_decimals(init_vals[:5], new_vals[:5], precision=2):          # REDUCE OR INCREASE THE PRECISION???
                break
            
            # save list of initialisation values obtained in a TXT file (additional information)
            # with open(output_path, "a") as file:
            #     file.write("\t".join(map(str, new_vals)) + "\n")
            # Update the initialisation values for the loop
            init_vals = new_vals.copy()
        #endwhile
        
        #convert history list to DF
        vallist = pd.DataFrame(history)
        
        #add on top of the report file the number of iteratios performed
        n_iters = len(history) - 1
        
        with open(output_path, "a") as file:
            file.write("Equilibrium reached after {:.0f} iterations\n".format(n_iters))
            file.write("DPM_tot\tRPM_tot\tBIO_tot\tHUM_tot\tIOM\tCO2\tTOC\n")
            vallist.to_csv(file, sep="\t", index=False, header=False)
        #endwith
        
        # CLOCK end actual spinup
        t_spinup_end = time.time()
        elapsed_spinup = t_spinup_end - t_spinup_start
        
        
        #%%               2ND STEP: compute the actual C fluxes for the actual DF
        
        # CLOCK start simulation with actual data
        t_sim_start = time.time()
        
        # compute RMF for the actual data using the model specified at the beginning of the script
        model_simulation = model
        df_sim = RMFs(df_sim, df_head, model_simulation)
        
        # get the last value that got out of the loop on the dummy DF
        initvals_post_eq = history[-1]
        # now compute C pools and SOC for the actual data
        df_actual, final_values = C_pools(df_sim, df_head, initvals_post_eq)
            
        # plot data now that the computations have been performed
        data = df_actual.iloc[1:].reset_index(drop=True).copy()
        data.insert(loc=2, column="time", value=pd.to_datetime(data[['year', 'month']].assign(day=1)))
        data['time'] = data['time'].dt.strftime('%m-%Y')
        
        
        #               SAVE DATA [values in MgC/ha]
        
        # save RAW DF with all data (even intermediate steps) for the actual years that had to be simulated
        data.to_csv(f'{output_folder}//RothC-AF_{filename_id}_out_DF_MgC_ha-1_RAW_data.csv', index=False)
        
        
        # save a simplified version of the output df with only the final values for the different pools & SOC
        df_simplified = data[['year', 'month', 'DPM_tot', 'RPM_tot', 'BIO_tot', 'HUM_tot', 'IOM', 'CO2_tot', 'TOC']].copy()
        
        # create a 'date' column in format 'yyyy-mm'
        df_simplified['year_month'] = (df_simplified['year'].round(0).astype(int).astype(str)
                                       + '-'
                                       + df_simplified['month'].round(0).astype(int).astype(str).str.zfill(2)
                                       )
        
        # reorder columns to put 'date' right after 'month'
        df_simplified = df_simplified[['year', 'month', 'year_month', 'DPM_tot', 'RPM_tot', 'BIO_tot', 'HUM_tot', 'IOM', 'CO2_tot', 'TOC']]
        
        # save DF to CSV
        df_simplified.to_csv(f'{output_folder}//RothC-AF_{filename_id}_out_DF_MgC_ha-1_simplified.csv', index=False)
        
        
        # from the simplified DF, create a new DF with the yearly cumulated values
        df_yearly = (df_simplified.groupby("year", as_index=False).agg({
            "DPM_tot": "sum",
            "RPM_tot": "sum",
            "BIO_tot": "sum",
            "HUM_tot": "sum",
            "CO2_tot": "sum",
            "IOM": "mean",   # IOM is constant, therefore a mea value can be set
            "TOC": ["mean", "sum"]    # change into "sum"/"mean"/"last" depending on the type of data you want
            }))
        # re-order column names
        # sistema i nomi delle colonne
        df_yearly.columns = ["year",
                             "DPM_cumul", "RPM_cumul", "BIO_cumul", "HUM_cumul", "IOM", 
                             "CO2_cumul", "TOC_mean", "TOC_cumul"]
        
        # save DF
        # df_yearly.to_csv(f"{output_folder}//RothC-AF_{filename_id}_out_DF_MgC_ha-1_yearly.csv", index=False)
        
        # create a copy with values converted from MgC/ha to kgC/m²
        conversion_cols = ['DPM_tot', 'RPM_tot', 'BIO_tot', 'HUM_tot', 'IOM', 'CO2_tot', 'TOC']
        df_converted = df_simplified.copy()
        df_converted[conversion_cols] = df_converted[conversion_cols] * 0.1
        
        # save DF to CSV    
        df_converted.to_csv(f'{output_folder}//RothC-AF_{filename_id}_out_DF_kgC_m-2_simplified.csv', index=False)
        
        # CLOCK end simulation with actual data
        t_sim_end = time.time()
        elapsed_sim = t_sim_end - t_sim_start
        
        # TOTAL duration of currently processed worksheet
        elapsed_total_sheet = t_sim_end - t_sheet_start

        # Report of processing times
        sheets_pbar.write(
            f"\n[✓ Field/Sheet: {sheet}]\n"
            f" ├── 1. C-Input optimization  : {elapsed_c_opt/60:.2f} min ({elapsed_c_opt:.2f} s) (Optimal C-input: {optimal_annual_c:.2f} MgC/ha)\n"
            f" ├── 2. Spin-up phase         : {elapsed_spinup/60:.2f} min ({elapsed_spinup:.2f} s) ({n_iters} iterations)\n"
            f" ├── 3. Actual simulation     : {elapsed_sim/60:.2f} min ({elapsed_sim:.2f} s)\n"
            f" └── Total time for sheet {sheet}: {elapsed_total_sheet/60:.2f} min ({elapsed_total_sheet:.2f} s)\n\n"
        )
        

    
        #%% Now plot data for the actual years to simulate
        
        
        # PLOT 1 - TOTAL SOC
        fig, ax = plt.subplots(figsize=(15, 8))
        
        #plot simulated SOC
        ln0 = ax.plot(data.time, data.TOC, c='k', label='Simulated TOC')
        
        #plot the Corg_init value as well
        init = ax.scatter(data.time.iloc[0], 
                         Corg_init,
                         color = 'red', 
                         s = 80,
                         zorder = 5,
                         label = 'Initial SOC'
                         )
        ln1 = [init]
        ax.set_xlabel('Time')
        ax.set_ylabel('t C ha-1')
        
        # plot climate variables on secondary axes to verify how SOC dynamics change in relation to T°/rain
        # these values must be taken from the df_sim variable, where only values for the actual simulated years are stored 
        # (without the average climate trend used for the spinup simulation)
        if len(data) != len(df_sim):
            sys.exit('ERROR: \nThe datasets "data" and "df_sim" must have the same size')
        else:
            df_clim = pd.DataFrame()
            df_clim["time"] = data.time
            df_clim["temp"] = df_sim.Temp
            df_clim["rain"] = df_sim.Prec
            df_clim["evap"] = df_sim.Evap
        #endif
        
        # 2nd axis for TEMPERATURE (lineplot)
        ax2 = ax.twinx()
        ln2 = ax2.plot(df_clim.time, df_clim.temp, ls='-.', color='red', label='Temperature')
        ax2.set_ylabel('°C', color='tab:red')
        
        # 3rd axis for RAINFALL (barplot)
        # ax3 = ax.twinx()
        # ax3.spines["right"].set_position(("axes", 1.08))
        # ln3 = ax2.plot(df_clim.time, df_clim.rain, color='blue', label='Rainfall + Irrigation')
        # ax3.set_ylabel('mm', color='tab:blue')
        
        # legend
        lns = ln0 + ln1 + ln2 #+ ln3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper right', fontsize=12)
        
        
        # axes formatter
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
        x_min, x_max = data['time'].min(), data['time'].max() 
        ax.set_xlim([x_min, x_max])
        ax2.set_xlim([x_min, x_max])
        # ax3.set_xlim([x_min, x_max])
        
        #SOC thresholds
        ax.set_ylim(bottom=data.TOC.min()-20, 
                    top=data.TOC.max()+20)  
        #temp thresholds
        ax2.set_ylim(bottom=df_clim.temp.min()-5, 
                     top=df_clim.temp.max()+5) 
        # ax3.set_ylim(bottom=0, top=200) #rain threhsolds
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')# plt.xticks(rotation=45)
        
        ax.set_title('Simulated SOC accumulation - {}'.format(filename_id), fontsize=15)
        
        #string w/info on initalization & equilibrium values
        textstr = (
            f'Measured SOC: {Corg_init:.2f} MgC/ha \n' 
            f'SOC at equilibrium: {init_vals[6]:.2f} MgC/ha \n'
            f'with yearly C-input = {float(df_eq.Herb_in.sum()/10) + float(df_eq.Tree_in.sum()/10):.2f} MgC/ha '
            )
        #add text to plot
        ax.text(
            0.02, 0.95,              # posizione relativa all'asse
            textstr,
            transform=ax.transAxes,
            fontsize=15,
            verticalalignment='top',
            bbox=dict(boxstyle='round',facecolor='white',alpha=0)
            )
        
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(output_folder, f'RothC-AF_{filename_id}_out_SOC.png'))
        
        #======================================================================
        
        # PLOT 2 - VALUES from different POOLS occurring to define TOC
        fig, ax = plt.subplots(figsize=(15, 8))
        
        #plot the different C pools
        ax.plot(data.time, data.DPM_tot, c='y', label='DPM')
        ax.plot(data.time, data.RPM_tot, c='green', label='RPM')
        ax.plot(data.time, data.BIO_tot, c='skyblue', label='BIO')
        ax.plot(data.time, data.HUM_tot, c='orange', label='HUM')
        ax.plot(data.time, data.IOM, c='blueviolet', label='IOM')
        ax.plot(data.time, data.TOC, c='black', ls='-.', label='TOC')
        
        plt.legend(fontsize=12, loc='upper right', ncols=6)
    
        ax.set_xlabel('Time')
        ax.set_ylabel('t C ha-1')
        ax.set_title('Carbon fluxes from different pools - {}'.format(filename_id), fontsize=15)
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
        ax.set_xlim([data['time'].min(), data['time'].max()])
        ax.set_ylim(bottom=0, top=data.TOC.max()+10)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(output_folder, f'RothC-AF_{filename_id}_out_Cpools.png'))
        
        #======================================================================
        
        # close all plots
        plt.close('all')
        
        # delete initialisation variables
        # del Corg_init, clay, depth
        
    #endfor
total_execution_time = time.time() - start_global_time
print("\n==========================================================================")
print("All files and worksheets have been processed")
print(f"Globale execution time: {total_execution_time / 60:.2f} minutes ({total_execution_time:.2f} s).")
print("==========================================================================\n")
#endfor



