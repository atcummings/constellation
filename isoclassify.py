# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 07:40:15 2023

@author: andre
"""

import astropy.units as u 
from astropy.table import Column 
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
import numpy as np 
import math 
import pandas as pd

direct_output = pd.read_csv('direct_output.csv')
direct_input = pd.read_csv('direct_input.csv')

'''direct_output = direct_output['id_starname',
'dir_dis', 'dir_dis_err1', 'dir_dis_err2', 'dir_avs', 'dir_avs_err1',
       'dir_avs_err2', 'dir_rad', 'dir_rad_err1', 'dir_rad_err2', 'dir_lum',
       'dir_lum_err1', 'dir_lum_err2', 'dir_teff', 'dir_teff_err1',
       'dir_teff_err2', 'dir_mabs', 'dir_mabs_err1', 'dir_mabs_err2',
       'dir_mass', 'dir_mass_err1', 'dir_mass_err2', 'dir_rho', 'dir_rho_err1',
       'dir_rho_err2', 'dir_fbol', 'dir_fbol_err1', 'dir_fbol_err2']'''

direct_output.replace(to_replace = -99, value = 0, inplace=True)


direct_output['ra'], direct_output['dec'] = 0,0
direct_input, direct_output = direct_input.set_index('id_starname'), direct_output.set_index('id_starname')

merged_data = pd.concat([direct_input,direct_output]).groupby(['id_starname']).sum()

merged_data['band'] = 'gamag'
merged_data['dust'] = ['allsky' if dec <= -30 else 'green19' for dec in merged_data['dec']]



merged_data = merged_data[[
 'DR2Name', 'ra', 'ra_err', 'dec', 'dec_err', 
 'HIP', 
'dir_dis', 'dir_dis_err1', 'dir_dis_err2', 'dir_avs', 'dir_avs_err1', 'dir_avs_err2', 'dir_rad', 'dir_rad_err1', 
'dir_rad_err2', 'dir_lum', 'dir_lum_err1', 'dir_lum_err2', 'dir_teff', 'dir_teff_err1', 'dir_teff_err2', 'dir_mabs', 
'dir_mabs_err1', 'dir_mabs_err2', 'dir_mass', 'dir_mass_err1', 'dir_mass_err2', 'dir_rho', 'dir_rho_err1', 'dir_rho_err2', 
'dir_fbol', 'dir_fbol_err1', 'dir_fbol_err2','dust','band','gamag','gamag_err']]

merged_data.rename(columns={'dir_dis':'dis', 'dir_dis_err1':'dis_err', 'dir_avs':'avs', 'dir_avs_err1':'avs_err',  'dir_rad':'rad', 'dir_rad_err1':'rad_err', 
'dir_lum':'lum', 'dir_lum_err1':'lum_err',  'dir_teff':'teff', 'dir_teff_err1':'teff_err', 'dir_mabs':'mabs', 
'dir_mabs_err1':'mabs_err',  'dir_mass':'mass', 'dir_mass_err1':'mass_err',  'dir_rho':'rho', 'dir_rho_err1':'rho_err',  
'dir_fbol':'fbol', 'dir_fbol_err1':'fbol_err'},inplace=True)



'''
'dmag', 'dmag_err','parallax', 'parallax_err', 'rplx', 'pmRA', 'e_pmRA', 'pmDE', 
'e_pmDE', 'gmag', 'gmag_err', 'FBP', 'e_FBP', 'RFBP', 'bpmag', 'bpmag_err', 
'FRP', 'e_FRP', 'RFRP', 'rpmag', 'rpmag_err', 'E_BR_RP_', 'Mode', 'BP-RP', 'BP-G', 
'G-RP', 'RV', 'e_RV', 'btmag', 'btmag_err', 'vtmag', 'vtmag_err', 'B-V', 'e_B-V', 'V-I', 
'e_V-I', 'bmag', 'bmag_err', 'vmag', 'vmag_err', 'umag', 'e_umag', 'gmag.1', 'e_gmag', 'rmag', 
'rmag_err', 'imag', 'imag_err', 'zmag', 'zmag_err', 'jmag', 'jmag_err', 'hmag', 'hmag_err', 'kmag', 
'kmag_err', 'hpmag', 'hpmag_err',
'''

'''merged_data = merged_data[(0.3 < merged_data['rad'] + merged_data['dir_rad_err2']) & 
                (merged_data['rad'] + merged_data['rad_err'] < 1.7)]'''

merged_data = merged_data[merged_data['teff']!=0 ]
merged_data = merged_data[merged_data['teff']>=4000] #FGK temp

merged_data = merged_data[merged_data['lum']!=0]
merged_data=merged_data.reset_index()
merged_data=merged_data.drop(columns=['lum','lum_err'])
merged_data.to_csv('grid_input.csv')
'''

#Use the merge function in pandas
def direct_merge(data,direct):
    merged_data = pd.merge(data, direct, on='id_starname', how='inner')

#Use the isin function
    data_index = data[data['id_starname'].isin(direct['id_starname'])].index
    direct_index = direct[direct['id_starname'].isin(data['id_starname'])].index

#Vectorize the operations of updating the dataframe
    data[["dir_dis", "dir_dis_err1", "dir_dis_err2", "dir_avs", 
      "dir_avs_err1", "dir_avs_err2", "dir_rad", "dir_rad_err1",
      "dir_rad_err2", "dir_lum", "dir_lum_err1", "dir_lum_err2", 
      "dir_teff", "dir_teff_err1", "dir_teff_err2", "dir_mabs", 
      "dir_mabs_err1", "dir_mabs_err2", "dir_mass", "dir_mass_err1", 
      "dir_mass_err2"]] = direct.loc[direct_index, ["dir_dis", 
        "dir_dis_err1", "dir_dis_err2", "dir_avs", "dir_avs_err1", 
        "dir_avs_err2", "dir_rad", "dir_rad_err1", "dir_rad_err2", 
        "dir_lum", "dir_lum_err1", "dir_lum_err2", "dir_teff", 
        "dir_teff_err1", "dir_teff_err2", "dir_mabs", "dir_mabs_err1",
        "dir_mabs_err2", "dir_mass", "dir_mass_err1", "dir_mass_err2"]]

    data.drop(data[data["dir_rad"] == 0].index, inplace=True)
    data.reset_index(drop=True)

    data = data[(0.3 < data['dir_rad'] + data['dir_rad_err2']) & 
                (data['dir_rad'] + data['dir_rad_err2'] < 1.7)]

    data['teff_err'] = data['dir_teff_err1']
    data.rename(columns={'dir_teff':'teff'},inplace=True)
    return data




grid_input = direct_merge(direct_input,direct_output)
grid_input.to_csv('grid_input.csv')

'''
