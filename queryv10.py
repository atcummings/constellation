# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:50:14 2023

@author: andre
"""

#Initial Star Query
from astropy import units as u
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from astroquery.mast import Catalogs
import numpy as np
import pandas as pd


#Queries HIP Catalog
catHip = 'I/239/hip_main'
v = Vizier(columns=['**'],column_filters={'Vmag':'<=4'}) #Select magnitude and band from HIP
v.ROW_LIMIT = -1
hip = v.query_constraints(catalog=catHip)[0]

#Queries GAIA DR2 Catalog
catGaia = 'I/345/gaia2'
vg = Vizier(catalog=catGaia, columns=['**'], column_filters={'Gmag':'<=4'}) #select magnitude and band from Gaia
vg.ROW_LIMIT=-1
gaia2 = vg.query_constraints(catalog=catGaia)[0]

#Queries TIC8 and avoids timeout error
tic = Catalogs.query_criteria(catalog="Tic", GAIAmag=[0,4], objType="STAR").to_pandas(index=True).sort_values(by=['ra']) 

'''
tic.rename(columns={'GAIA':'DR2Name','plx':'parallax','e_plx':'parallax_err','Bmag':'bmag','e_Bmag':'bmag_err','Vmag':'vmag','e_Vmag':'vmag_err',
					'umag':'umag','e_umag':'umag_err','gmag':'gmag','e_gmag':'gmag_err','rmag':'rmag','e_rmag':'rmag_err','imag':'imag','e_imag':'imag_err',
					'zmag':'zmag','e_zmag':'zmag_err','Jmag':'jmag','e_Jmag':'jmag_err','Hmag':'hmag','e_Hmag':'hmag_err','Kmag':'kmag','e_Kmag':'kmag_err',
					'GAIAmag':'gamag','e_GAIAmag':'gamag_err','Tmag':'tmag','e_Tmag':'tmag_err','gaiabp':'bpmag','e_gaiabp':'bpmag_err','gaiarp':'rpmag',
					'e_gaiarp':'rpmag_err','RAJ2000':'ra','DEJ2000':'dec'},inplace=True)

tic = tic[['DR2Name','HIP','parallax','parallax_err','bmag','bmag_err','vmag','vmag_err',
					'umag','umag_err','gmag','gmag_err','rmag','rmag_err','imag','imag_err',
					'zmag','zmag_err','jmag','jmag_err','hmag','hmag_err','kmag','kmag_err',
					'gamag','gamag_err','tmag','tmag_err','bpmag','bpmag_err','rpmag',
					'rpmag_err','ra','dec']]
'''

hip_temp=hip.to_pandas(index=True)
gaia2_temp=gaia2.to_pandas(index=True)

hip_temp = hip_temp.reindex(columns=['HIP','BTmag','e_BTmag','VTmag','e_VTmag','B-V','e_B-V','V-I','e_V-I',
    'Hpmag','e_Hpmag','GAIA','Bmag', 'e_Bmag', 'Vmag', 'e_Vmag', 'umag', 'e_umag', 
    'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 
    'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag','GAIAmag','e_GAIAmag'])




tic = tic.reindex(columns=[
    'HIP','GAIA','Bmag', 'e_Bmag', 'Vmag', 'e_Vmag', 'umag', 'e_umag', 
    'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 
    'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag'
    'BTmag','e_BTmag','VTmag','e_VTmag','B-V','e_B-V','V-I','e_V-I',
    'Hpmag','e_Hpmag','GAIAmag','e_GAIAmag'
    ])

tic['HIP'], hip_temp['HIP'] = tic['HIP'].astype(float), hip_temp['HIP'].astype(float)
tic['GAIA'], hip_temp['GAIA'] = tic['GAIA'].astype(float), hip_temp['GAIA'].astype(float)


tic, hip_temp = tic.set_index('HIP'), hip_temp.set_index('HIP')

tic_hip_merge = pd.concat([tic,hip_temp]).groupby(['HIP']).sum()

tic_hip_merge = tic_hip_merge.reset_index()

tic_hip_merge = tic_hip_merge.reindex(columns=[
'HIP','GAIA','Bmag', 'e_Bmag', 'Vmag', 'e_Vmag', 'umag', 'e_umag', 
    'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 
    'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag', 'RAJ2000','e_RAJ2000','DEJ2000','e_DEJ2000',
    'Plx','e_Plx','RPlx','pmRA','e_pmRA','pmDE','e_pmDE',
    'Gmag','e_Gmag','FBP','e_FBP','RFBP','BPmag','e_BPmag',
    'FRP','e_FRP','RFRP','RPmag','e_RPmag','E_BR_RP_','Mode','BP-RP','BP-G',
    'G-RP','RV','e_RV','BTmag','e_BTmag','VTmag','e_VTmag','B-V','e_B-V','V-I','e_V-I',
    'Hpmag','e_Hpmag','GAIAmag','e_GAIAmag'
    ])

tic_hip_merge = tic_hip_merge.rename(columns={'GAIA':'DR2Name'})
gaia2_temp = gaia2_temp.reindex(columns=[
    'DR2Name','RAJ2000','e_RAJ2000','DEJ2000','e_DEJ2000',
    'Plx','e_Plx','RPlx','pmRA','e_pmRA','pmDE','e_pmDE',
    'Gmag','e_Gmag','FBP','e_FBP','RFBP','BPmag','e_BPmag',
    'FRP','e_FRP','RFRP','RPmag','e_RPmag','E_BR_RP_','Mode','BP-RP','BP-G',
    'G-RP','RV','e_RV','BTmag','e_BTmag','VTmag','e_VTmag','B-V','e_B-V','V-I','e_V-I',
    'Bmag', 'e_Bmag', 'Vmag', 'e_Vmag', 'umag', 'e_umag', 
    'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag', 
    'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag',
    'Hpmag','e_Hpmag','GAIAmag','e_GAIAmag'])


for i in gaia2_temp.index: 
	gaia2_temp['DR2Name'][i]= gaia2_temp['DR2Name'][i][9:]

gaia2_temp['DR2Name'] = gaia2_temp['DR2Name'].astype(float)

tic_hip_merge, gaia2_temp = tic_hip_merge.set_index('DR2Name'), gaia2_temp.set_index('DR2Name')

starlist = pd.concat([gaia2_temp,tic_hip_merge]).groupby(['DR2Name']).sum()
starlist = starlist.tail(-1).reset_index()

starlist =starlist[starlist['B-V']>=0.19]

starlist[['Plx', 'e_Plx']] = starlist[['Plx', 'e_Plx']] / 1000
starlist['id_starname'] = 0
starlist['id_starname'] = 'star_' + (starlist.index + 1).astype(str)

starlist.rename(columns={'RAJ2000':'ra', 'e_RAJ2000':'ra_err', 'DEJ2000':'dec', 'e_DEJ2000':'dec_err', 'Plx':'plx',
       'e_Plx':'plx_err', 'RPlx':'rplx',  'Gmag':'gmag', 'e_Gmag':'gmag_err',
       'BPmag':'bpmag', 'e_BPmag':'bpmag_err', 
       'RPmag':'rpmag', 'e_RPmag':'rpmag_err','BTmag':'btmag', 'e_BTmag':'btmag_err', 'VTmag':'vtmag', 'e_VTmag':'vtmag_err',
       'Bmag':'bmag', 'e_Bmag':'bmag_err', 'Vmag':'vmag', 'e_Vmag':'vmag_err',  'Jmag':'jmag',
       'e_Jmag':'jmag_err', 'Hmag':'hmag', 'e_Hmag':'hmag_err', 'Kmag':'kmag', 'e_Kmag':'kmag_err', 'Hpmag':'hpmag', 'e_Hpmag':'hpmag_err',
      'e_rmag':'rmag_err', 'e_imag':'imag_err','e_zmag':'zmag_err','GAIAmag':'gamag','e_GAIAmag':'gamag_err'
       

    },inplace=True)

starlist['dust'] = ['allsky' if dec <= -30 else 'green19' for dec in starlist['dec']]


starlist['band'] = 'vtmag'



starlist['id_starname'].astype('object') #Isoclassify likes these columns as objects
starlist['dust'].astype('object') 
starlist['band'].astype('object') 

starlist.replace(to_replace = 0, value = -99, inplace=True)
'''starlist = starlist[['DR2Name', 'ra', 'ra_err', 'dec', 'dec_err', 'plx', 'plx_err', 
       'btmag', 'btmag_err', 'vtmag', 'vtmag_err', 
        'HIP', 'id_starname', 'dust', 'band']]'''
starlist.rename(columns={'plx':'parallax','plx_err':'parallax_err'},inplace=True)
starlist['teff'] = -99.0
starlist['teff_err']=0

starlist['lum']=-99.0
starlist['lum_err']=0

starlist['rad']=-99.0
starlist['rad_err']=0
starlist['feh']=-99.0
starlist['feh_err']=0
starlist['logg']=-99.0
starlist['logg_err']=0

starlist.to_csv('direct_input.csv')