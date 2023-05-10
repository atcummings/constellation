import astropy.units as u 
from astropy.table import Column 
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
import numpy as np 
import math 
import pandas as pd

direct_output = pd.read_csv('direct_output.csv')
direct_input = pd.read_csv('direct_input.csv')

def grid_merge(data,grid):

    merged_df = pd.merge(data, grid, on='id_starname', how='inner')

    data_index = data[data['id_starname'].isin(grid['id_starname'])].index
    grid_index = grid[grid['id_starname'].isin(data['id_starname'])].index

    data[["iso_age", "iso_age_err2", "iso_avs", 
                      "iso_avs_err1", "iso_avs_err2", "iso_dis", 
                      "iso_dis_err1", "iso_dis_err2", "iso_feh", 
                      "iso_feh_err1", "iso_feh_err2", "iso_mass", 
                      "iso_mass_err1", "iso_mass_err2", "iso_rad", 
                      "iso_rad_err1", "iso_rad_err2", "iso_lum", 
                      "iso_lum_err1", "iso_lum_err2", "iso_logg", 
                      "iso_logg_err1", "iso_logg_err2", "iso_teff", 
                      "iso_teff_err1", "iso_teff_err2"]] = grid.loc[grid_index, 
        ["iso_age", "iso_age_err2", "iso_avs", 
                          "iso_avs_err1", "iso_avs_err2", "iso_dis", 
                          "iso_dis_err1", "iso_dis_err2", "iso_feh", 
                          "iso_feh_err1", "iso_feh_err2", "iso_mass", 
                          "iso_mass_err1", "iso_mass_err2", "iso_rad", 
                          "iso_rad_err1", "iso_rad_err2", "iso_lum", 
                          "iso_lum_err1", "iso_lum_err2", "iso_logg", 
                          "iso_logg_err1", "iso_logg_err2", "iso_teff", 
                          "iso_teff_err1", "iso_teff_err2"]]

    '''cols_to_update = ["iso_age", "iso_age_err2", "iso_avs", 
                      "iso_avs_err1", "iso_avs_err2", "iso_dis", 
                      "iso_dis_err1", "iso_dis_err2", "iso_feh", 
                      "iso_feh_err1", "iso_feh_err2", "iso_mass", 
                      "iso_mass_err1", "iso_mass_err2", "iso_rad", 
                      "iso_rad_err1", "iso_rad_err2", "iso_lum", 
                      "iso_lum_err1", "iso_lum_err2", "iso_logg", 
                      "iso_logg_err1", "iso_logg_err2", "iso_teff", 
                      "iso_teff_err1", "iso_teff_err2"]'''


    #data.update(merged_df[cols_to_update])
    
    data['id_starname'] = data['id_starname'].str[7:]
    data = data[(0.7 < data['iso_rad'] + data['iso_rad_err2']) & 
                (data['iso_rad'] + data['iso_rad_err2'] < 1.4)]
    
    return data

final_star_list = grid_merge(grid_input, grid_output)
final_star_list.to_csv('final_star_list.csv')