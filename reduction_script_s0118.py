#!/usr/bin/env python
# coding: utf-8

# # Script infors

# Version 0.7    
# script to read and analyze experimental data from the ***biassiale*** deformation apparatus  
# here I report only the pandas approach and I make the script a little more easy to use.  
# - very new in this version is the implementation in rawPy of a load_tdms that allows to easily open a data file

# our data are saved in tdms format  
# To read the file, the .tdms file and the .tdms_index filenshould be in the same directory.   
# To open the file in Python we use the [**npTDMS**](https://nptdms.readthedocs.io/en/stable/index.html) library that can be installed:  
# **pip install nptdms**

# # Insert experiment infos

# ##########################  
# **Date Run:** 
# **Date Reduced:** 
# ##########################  
# **Material:** 
# **Configuration:** Double-Direct Shear 5x5 5x5 small grooves side blocks  
# **Normal stress:** 
# **Shear velocity:**  
# **Velocity steps:** 
# **Slide-hold-slide:** 
# **Notes:**  
# W1 = 
# W2 = 
# In[1]:
###################### CALIBRATION FUNCTIONS
def horizontal_calibration(volt):
    '''
    input:
        Voltage (must be offsetted)
    Return: 
        Force (KN)
    '''
    x=np.array(volt)
    coefficients = np.array([-4.63355231e-02, -2.57055418e+00,  2.63055688e+01, -9.61932787e+01,
            1.64685122e+02, -1.33648859e+02,  4.66773182e+01,  1.63975941e+02,
            9.32438525e-02])
    hor_calibration = np.poly1d(coefficients)
    force = hor_calibration(x)   
    return force

def vertical_calibration(volt):
    '''
    input:
        Voltage (must be offsetted)
    Return: 
        Force (KN)
    '''
    x=np.array(volt)
    coefficients = np.array([ -0.5043737 ,   4.27584024, -11.70546934,   5.45745069,
            29.43390033, -60.90428874,  60.98729795, 124.19783947,
            -0.47000267])
    vert_calibration = np.poly1d(coefficients)
    force = vert_calibration( x)
    return force
######################## ELASTIC CORRECTION FUNCTIONS
def VerticalStiffness (force):
    '''
    input:
        Force (in KN)
    Return: 
        Stiffness array
    '''
    coefficients = np.array([ 3.35241499e-30, -9.37367134e-27,  1.19440060e-23, -9.17845027e-21,
            4.74088379e-18, -1.73500082e-15,  4.61575497e-13, -9.00528796e-11,
            1.28295415e-08, -1.31327065e-06,  9.38895324e-05, -4.50176164e-03,
            1.38008389e-01, -2.63525139e+00,  3.57750394e+01,  1.71503762e+01])
    calibration = np.poly1d(coefficients)
    stiffness = calibration(force)
    return stiffness

def HorizontalStiffness (force):
    '''
    input:
        Force (in KN)
    Return: 
        Stiffness array
    '''
    coefficients = np.array([ 2.43021220e-31, -7.73507440e-28,  1.10791696e-24, -9.43050473e-22,
            5.30556343e-19, -2.07533887e-16,  5.77817817e-14, -1.15148744e-11,
            1.62528123e-09, -1.57483543e-07,  9.75756659e-06, -3.16390679e-04,
            1.96801181e-04,  2.69515293e-01,  5.53939566e+00,  4.21560673e+01])
    calibration = np.poly1d(coefficients)
    stiffness = calibration(force)
    return stiffness


def NonlinearElasticCorrection(stress,disp,k):
    '''
    input:
        force
        disp
        k = stiffness
    Return: 
        elestic corrected displacement
    '''
    from pandas.core.series import Series
    k = k[:-1]
    # Type conversion from Pandas to NumPy
    if type(stress) == type(Series()):
        stress = stress.values
    if type(disp) == type(Series()):
        disp = disp.values    
    # Increments in elastic distortion
    dload = (stress[1:] - stress[:-1]) / k
    # Increments in total displacement
    ddisp = disp[1:] - disp[:-1]
    # Subtract elastic distortion from total displacement
    ec_disp = np.hstack([0, np.cumsum(ddisp - dload)])
    return ec_disp

import os
import numpy as np
import pandas as pd
from rawPy.rawPy import rawPy as rp
from scipy.stats import linregress
import matplotlib.pyplot as plt 


# # 1. Open data file  

# - the reduction script and the data file should be in the same directory 
# - The functio load_tdms returns a pandas DataFrame 

# In[2]:
%matplotlib qt

### inpu the name of the experiment ###
exp_name = 'cal_loadcell'
#######################################

path = os.getcwd()
df = rp.load_tdms('%s/%s.tdms'%(path,exp_name))
df

#%%
calibration_v = 1 # insert the calibration value used
calibration_h = 1 # insert the calibration value used
v_volt = df.Vertical_Load/calibration_v
n_volt = df.Horizontal_Load/calibration_h

#plt.plot (df.Vertical_Load,'b')
vertical_load = vertical_calibration(v_volt-v_volt[0])
vertical_load = vertical_load - vertical_load[0]

horizontal_load = horizontal_calibration(n_volt-n_volt[0])
horizontal_load = horizontal_load + horizontal_load[0]

# In[9]:plt

# # 3. Vertical piston

# In[3]:
# Verifica delle colonne esistenti nel DataFrame
required_columns = ['Rec_n', 'Horizontal_Load', 'Vertical_Displacement', 'Time']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

# Verifica delle lunghezze delle serie/liste
records_na = df['Rec_n']
time_s = df['Time']
horizontal_load = df['Horizontal_Load']
vertical_displacement = df['Vertical_Displacement']

print(f"Length of vertical_load: {len(vertical_load)}")
print(f"Length of records_na: {len(records_na)}")
print(f"Length of time_s: {len(time_s)}")
print(f"Length of horizontal_load: {len(horizontal_load)}")
print(f"Length of vertical_displacement: {len(vertical_displacement)}")

# Assicurati che tutte le lunghezze siano uguali
if not (len(vertical_load) == len(records_na) == len(time_s) == len(horizontal_load) == len(vertical_displacement)):
    raise ValueError("All lists/series must have the same length")

data_out = [
    vertical_load,
    horizontal_load,
    vertical_displacement,
    time_s,
    records_na
]

rp.save_data(exp_name, data_out, callingLocals=locals())


# In[7]:
#### Vertical stress ###
#### This is the point at which the ram contacted the blocks and load shear loading began

beg_row_v = 1790
##################################

# ZERO DATA
v_load = rp.zero(vertical_load,beg_row_v)
# add in force from the central block due to gravity 
# v_load = v_load + 0.044 #[kN]
# remove noise before load is applied
v_load[:beg_row_v]= v_load[:beg_row_v]*0
# calculate stress for DDS 5x5
shear_stress_MPa = v_load/(2*1000*0.0025)
#del v_load
fig=plt.figure(figsize=(15,6))
plt.plot(df.Time, shear_stress_MPa)
plt.ylabel('Shear stress [MPa]')
plt.xlabel('Time [s]')
plt.tight_layout()

# ### 3. Correct for elastic strech of the vertical frame   
# 
# | Verical stiffness | applied load | k[MPa/mm] | calibration date | 
# | --- | --- | --- | --- |  
# | 359.75 [kN/mm] | < 50 [kN] | 116.801 [MPa/mm] | 19/11/2015 | 
# | 928.5 [kN/mm] | > 50 [kN] | 301.461 [MPa/mm] | 19/11/2015 | 
# 
# Note:  
# 1. These values are calculated for a 5x5[cm] shear surface  
# 2. Stiffness is non linear at low applied loads  
# TO DO: implement a function that corrects for non-linear stiffness
# 

# In[9]:
vertical_force = []
for i in range (0,len(v_load)):
    if v_load[i]>0:
        vertical_force.append(v_load[i])
    else:
        vertical_force.append(0)
vertical_force = np.array(vertical_force)
#### vertical disp ####
lp_disp_mm = rp.zero(df.Vertical_Displacement,beg_row_v) #to have it it mm
lp_disp_mm[:beg_row_v] = lp_disp_mm[:beg_row_v]*0
lp_disp_mm = lp_disp_mm/1000

rp.plot(df.Time,shear_stress_MPa,'Time [s]','Shear stress [MPa]')
rp.plot(df.Time,lp_disp_mm,'Time [s]', 'lp_disp [mm]')

# Elastic correction for the stretch of the vertical frame # 
k_constant = 166.801 #[MPa/mm]
k = VerticalStiffness(vertical_force) #
#print(k)

ec_disp_mm = NonlinearElasticCorrection(vertical_force,lp_disp_mm,k) 
ec_disp_mm_old = rp.ElasticCorrection(shear_stress_MPa,lp_disp_mm,k_constant)

# OFFSET data 
# takes as input row1, row2, col 
#example
#row_1 = 89726
#row_2 = 89737
#lp_disp_mm = rp.offset(lp_disp_mm,row_1,row_2)
#


# # 4. Horizontal Piston 

# Convert force to load 

# In[10]:
plt.figure()
plt.plot(lp_disp_mm,shear_stress_MPa,label='not corrected')
plt.plot(ec_disp_mm,shear_stress_MPa,label='new correction')
plt.plot(ec_disp_mm_old,shear_stress_MPa,label='old correction')
plt.xlabel('Corrected load point displacement (mm)')
plt.ylabel('Shear stress (MPa)')

plt.xlim([0,20])

plt.legend()

#%%

hor_load = df.Horizontal_Load-df.Horizontal_Load[0]
rp.plot(df.Rec_n, hor_load)

# In[12]:


#############  normal load ############ 
#### Normal load is applied at record

%matplotlib qt

beg_row_h = 136

######################################
# zero the data
h_load = rp.zero(horizontal_load,beg_row_h)
# remove noise before load is applied 
h_load[:beg_row_h] = h_load[:beg_row_h]*0
# calculate stress for DDS 5x5
h_load =  h_load/(1000*0.0025)
#add a small number to normal stress 
#so that we aren't dividing by 0 anywhere to calculate mu 
normal_stress_MPa = h_load + 1e-7

rp.plot(df.Time,normal_stress_MPa,'Time [s]','Normal stress [MPa]')

# ### 4. Correct for elastic strech of the horizontal frame   
# 
# | Horizontal stiffness | applied load | k[MPa/mm] | calibration date | 
# | --- | --- | --- | --- |  
# | 386.12 [kN/mm] | < 50 [kN] | 125.363 [MPa/mm] | 19/11/2015 | 
# | 1283 [kN/mm] | > 50 [kN] | 416.558 [MPa/mm] | 19/11/2015 | 
# 
# Note:  
# 1. These values are calculated for a 5x5[cm] shear surface  
# 2. Stiffness is non linear at low applied loads  
# TO DO: implement a function that corrects for non-linear stiffness
# 

# In[14]:
horizontal_force = []
for i in range (0,len(h_load)):
    if h_load[i]>0:
        horizontal_force.append(h_load[i])
    else:
        horizontal_force.append(0)
horizontal_force = np.array(horizontal_force)


# Elastic correction for the stretch of the horizontal frame #
k_constant = 125.363 #[MPa/mm]
k = HorizontalStiffness(horizontal_force)

lt_ec_mm = NonlinearElasticCorrection(horizontal_force,-df.Horizontal_Displacement/1000,k)
lt_ec_mm_old = rp.ElasticCorrection(normal_stress_MPa,-df.Horizontal_Displacement/1000,k_constant)


# # 5. Calculate layer thickness   

# 
# Treat changes in horizontal displacement ($\Delta$h) as symmetric, take half of it for 1 layer.  
# Compaction = thinner layer  
# Thickness of DDS assembly with no gouge is:  
# 
# | small groove blocks | large groove blocks | PZT side blocks |  
# | --- | --- | --- |  
# |99.7 mm | 95 mm | 102.54 mm |
# 
# Bench thickness of initial layers & DDS is  --- [mm]  
# Bench thickness of 1 layer is: ---  [mm]
# 
# Thickness of spacers:  
# 
# | AA | BB | CC |  
# | --- | --- | --- |  
# |29.86 mm | -- mm | 6.12 mm |
# 
# The layer thickness is calculated as:  
# layer_thickness = (total layer thickness)-(Assembly)-(Spacers) / 2  
# 
# 
# Total thickness is 137.14 mm at rec #6068  
# layer tickness is  (137.14- 99.7 -29.86-6.12)/2  
# Layer thickness for one layer under load is 0.73 mm  

# In[15]:


###########################################################
# insert rec number at which layer thickness was measured
rec_lt = 136
# insert calculated value at that point
val_lt = 5
###########################################################
# zero data
lt_ec_mm = rp.zero(lt_ec_mm,rec_lt)
lt_ec_mm_half = lt_ec_mm/2
lt_ec_mm1 = lt_ec_mm_half + val_lt

lt_ec_mm_old = rp.zero(lt_ec_mm_old,rec_lt)
lt_ec_mm_old_half = lt_ec_mm_old/2
lt_ec_mm_old1 = lt_ec_mm_old_half + val_lt


rp.plot(df.Time,lt_ec_mm1)
rp.plot(df.Time,lt_ec_mm_old1)


# ### Remove geometrical thinning

# In[ ]:
# default unit is [mm]
rgt_lt_mm = rp.rgt(lp_disp_mm,lt_ec_mm1)
# # 6. Calculate friction 

# In[ ]:
##### insert the last row 
end_row = 302178
########################

friction_na = shear_stress_MPa/normal_stress_MPa
# remove the noise before and after vertical load
friction_na[:beg_row_v] = friction_na[:beg_row_v]*0
friction_na[end_row:] = friction_na[end_row:]*0

# # 7. Calculate shear strain 

# In[ ]:

##### calculate strain ####
shear_strain_na = rp.shear_strain(ec_disp_mm,lt_ec_mm1)

plt.figure()
plt.plot(ec_disp_mm,friction_na)
plt.xlabel('Displacement (mm)',fontsize=20)
plt.ylabel('Friction',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# # 8. Export the data
# In[ ]:

data_out = [shear_stress_MPa,
            lp_disp_mm,
            ec_disp_mm, 
            normal_stress_MPa,
            lt_ec_mm1, 
            rgt_lt_mm,
            friction_na,
            time_s,
            records_na
          ]
            
rp.save_data(exp_name,data_out,callingLocals=locals())


# %%
# %%
exp_name
# %%
