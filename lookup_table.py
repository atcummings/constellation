# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 07:20:49 2023
# -*- coding: utf-8 -*-

Created on Wed Feb  8 09:36:00 2023

@author: andre
"""


#testcase to create 3 satellites.
#from SimSupport import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import astropy.stats
import emcee
import scipy.optimize



#data = np.asarray(pd.read_csv('~/simseismo-master/Tau1Eri/ts_noise_90_TauEri.csv')).T


from scipy.special import factorial
from scipy.interpolate import interp1d

lmax = 2


def data_sampler(time, flux, num_samples=10):
    # Create DataFrame
    df = pd.DataFrame({'time': time, 'flux': flux})

    # Compute phase of each data point
    df['phase'] = (df['time'] % 1) * 10

    # Define phase bins
    bins = pd.cut(df['phase'], 10, labels=False)

    # Downsample each phase bin separately
    stratified = []
    for i in range(10):
        group = df[bins == i]
        if len(group) < num_samples:
            num_samples_group = len(group)
        else:
            num_samples_group = num_samples
        sample = group.sample(num_samples_group, replace=False)
        stratified.append(sample)

    # Combine samples from all bins
    stratified = pd.concat(stratified)

    # Extract downsampled time and flux arrays
    downsampled_time = stratified['time'].values
    downsampled_flux = stratified['flux'].values

    # Return downsampled data
    return downsampled_time, downsampled_flux



# Define the limb darkening function
def limb_darkening(u, a=0.6):
    return 1.0 - a * (1.0 - u)

# Define the spherical harmonics function
def spherical_harmonics(theta, phi, l, m):
    pmm = 1.0
    if m > 0:
        fact = 1.0
        for i in range(1, m+1):
            pmm *= -fact * np.sin(theta)**2
            fact += 2.0
    if m < 0:
        m = -m
        pmm = (factorial(2*m-1) / factorial(2*m))**0.5 * np.sin(theta)**m * np.exp(1j * m * phi)
    if l == m:
        return pmm
    else:
        pmmp1 = np.cos(theta) * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1
        else:
            pll = 0.0
            for i in range(m+2, l+1):
                pll = ((2.0 * i - 1.0) * np.cos(theta) * pmmp1 - (i + m - 1.0) * pmm) / (i - m)
                pmm = pmmp1
                pmmp1 = pll
            return pll

# Define the log likelihood function for the MCMC
def log_likelihood(theta, flux, times, lmax):
    incl, a = theta
    y = np.zeros_like(flux)
    for l in range(lmax+1):
        for m in range(-l, l+1):
            for i in range(len(flux)):
                u = np.sin(incl) * np.cos(times[i]) + np.cos(incl) * np.sin(times[i]) * np.sin(np.arctan2(np.sqrt(1 - np.sin(incl)**2), np.cos(incl)))
                y[i] += a * spherical_harmonics(np.arccos(u), 0.0, l, m).real * limb_darkening(u, a) * \
                    np.exp(-(l*(l+1)) * (1 - np.cos(np.arccos(u))))
    return -0.5 * np.sum((flux - y)**2)

# Define the log prior function for the MCMC
def log_prior(theta):
    incl, a = theta
    if 0.0 <= incl <= np.pi/2.0 and 0.0 <= a <= 1.0:
        return 0.0
    else:
        return -np.inf

# Define the log probability function for the MCMC
def log_probability(theta, flux, times, lmax):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, flux, times, lmax)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# Define the function that finds the inclination using MCMC
def find_inclination(flux, times, lmax, nwalkers=32, nsteps=200):
    # Initialize the MCMC sampler
    ndim = 2
    p0 = np.random.rand(nwalkers, ndim)
    p0[:, 0] *= np.pi/2.0
    p0[:, 1] *= 1.0
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(flux, times, lmax))

# Run the MCMC sampler
    pos, prob, state = sampler.run_mcmc(p0, nsteps)

# Discard the first half of the samples as burn-in
    sampler.reset()
    sampler.run_mcmc(pos, nsteps,progress=True)

# Get the posterior distribution of the inclination
    incl_samples = sampler.flatchain[:, 0]
    incl_posterior = np.histogram(incl_samples, bins=100, density=True)

# Find the maximum of the posterior distribution
    incl_max = incl_posterior[1][np.argmax(incl_posterior[0])]
    return np.degrees(incl_max)


### MCMC inclination code end

def mask(start,stop, lightcurve,loop_var):
    #stk_sim = pd.read_csv(stk_sim)
    #lightcurve = np.asarray(pd.read_csv(lightcurve)).T

    #t_lightcurve = lightcurve[0] * 3600 * 24
    f_lightcurve = lightcurve[1] # in ppm
    # assuming you have an array of flux values called "flux_array" and a loop variable called "loop_var"
    if loop_var > 1:
    # calculate the integer part and fractional part of the loop variable
        int_part = int(loop_var)
        frac_part = loop_var - int_part
    else:
        int_part = 0
        frac_part = loop_var
    
# concatenate the first k elements of the flux array to the end, where k is the fractional part of the loop variable
    k = int(len(f_lightcurve) * frac_part)
    if k > 0:
        f_lightcurve = np.hstack((f_lightcurve, f_lightcurve[:k]))
    if int_part > 0:
        for i in range(int_part - 1):
            f_lightcurve = np.hstack((f_lightcurve, f_lightcurve[:]))
    
    t_start = 30
    t_end = t_start + 30 * (len(f_lightcurve) - 1)
    t_lightcurve = np.arange(t_start, t_end + 1, 30)


    #start = stk_sim['start times'] #[0]
    #stop = stk_sim['stop times'] #[0]
    start = start
    stop = stop

    def create_mask(t_lightcurve, start, stop):
        mask = np.zeros(t_lightcurve.shape, dtype=bool)
        for i in range(len(start)):
            mask |= (t_lightcurve >= start[i]) & (t_lightcurve <= stop[i])
        return mask

    mask = create_mask(t_lightcurve, start, stop)
    indices = np.where(mask)
    '''
    plt.scatter(t_lightcurve[indices], f_lightcurve[indices],
                s=0.5, color='red', zorder=10, label='Mask')
    plt.scatter(t_lightcurve, f_lightcurve, s=0.5, color='black', zorder=1, label='Synthetic Lightcurve')
    plt.show()
    '''
    data = pd.DataFrame({'time':t_lightcurve[indices],
                         'flux':f_lightcurve[indices]})
    
    return data

### percent Error code
def percent_error(observed_value, error_bar, expected_value):
    percent_error_upper = abs((observed_value + error_bar - expected_value) / expected_value) * 100
    percent_error_lower = abs((observed_value - error_bar - expected_value) / expected_value) * 100
    average_percent_error = (percent_error_upper + percent_error_lower) / 2
    return average_percent_error



try:
    if os.name == "nt":
        from agi.stk12.stkdesktop import STKDesktop
    from agi.stk12.stkobjects import *
    from agi.stk12.stkutil import *
    from agi.stk12.utilities.colors import *
except:
    print("Failed to import stk modules. Make sure you have installed the STK Python API wheel (agi.stk<..ver..>-py3-none-any.whl) from the STK Install bin directory")
if os.name == "nt":
    app = STKDesktop.StartApplication(visible=False, userControl=True)
    ObjectRoot = app.Root
else:
    print("Automation samples only work on windows with the desktop application, see Custom Applications for STK Engine examples.")
    quit()

#Specify the Star List

'''
def TrueAnomalySpacing(value,NumberOfSatellites):
    if value == 'even' or 'Even' == True:
        Spacing = np.linspace(0,360,NumberOfSatellites,endpoint=False)
    if value != 'even' or 'Even' == True:
        Spacing = np.linspace(0,NumberOfSatellites*value,NumberOfSatellites)
    return Spacing
'''


def TrueAnomalySpacing(value, NumberOfSatellites):
    if value.lower() == 'even':
        Spacing = np.linspace(0, 360, int(NumberOfSatellites), endpoint=False)
    elif value.isnumeric():
        Spacing = np.linspace(0, int(NumberOfSatellites) * float(value), NumberOfSatellites)
    else:
        raise ValueError("Invalid value for 'value' parameter.")
    return Spacing


StarList = 'target_star_list.csv'

data_dir = "star_data/"

# Load the master CSV file
StarList = pd.read_csv('target_star_list.csv')
StarList['name'] = StarList['name'].astype(str)
# Load the ts_noise.csv files for each subdirectory
ts_noise_files = {}
for subdir in StarList["name"]:
    file_path = os.path.join(data_dir, subdir, "ts_noise.csv")
    ts_noise_files[subdir] = np.asarray(pd.read_csv(file_path)).T








def simulation(x,ts_noise_files):
    
   StarList = pd.read_csv('target_star_list.csv')
   StarList['name'] = StarList['name'].astype(str)
   
   #StarList = 'target_star_list.csv'#'teststarlist.csv' #
   
   Nsatellites=x[0]
   MissionDuration=1#x[1]
   Inclination=x[2]
   Altitude=x[1]
   ''' where:
       x1 = Nsatellites [unitless]
       x2 = MissionDuration [year]
       x3 = AltSat [km]
       x4 = IncSat [degree 0-90]
       '''
       
   SimStart = "1 Jul 2002 00:00:00.00"
   start_time = datetime.strptime(SimStart, "%d %b %Y %H:%M:%S.%f")
   time_delta = MissionDuration*timedelta(days=365)
   duration = start_time+time_delta
   #this is a new way to specify the time duration of the entire simulation
   SimEnd = duration.strftime("%d %b %Y %H:%M:%S.%f")
   
   TrueAnomalySpace = 'even'
    
   print("New Scenario...")
   ObjectRoot.NewScenario("Constellation")

   dimensions = ObjectRoot.UnitPreferences
   dimensions.ResetUnits()
   dimensions.SetCurrentUnit("DateFormat", "UTCG")
   scene = ObjectRoot.CurrentScenario

   scene.StartTime = SimStart
   scene.StopTime = SimEnd
   scene.Epoch = SimStart



    # Define a dictionary containing the units and their values
   units = {"DistanceUnit": "km", "TimeUnit": "sec", "AngleUnit": "deg",
             "MassUnit": "kg", "PowerUnit": "dbw", "FrequencyUnit": "ghz",
             "SmallDistanceUnit": "m", "latitudeUnit": "deg", "longitudeunit": "deg",
             "DurationUnit": "HMS", "Temperature": "K", "SmallTimeUnit": "sec",
             "RatioUnit": "db", "rcsUnit": "dbsm", "DopplerVelocityUnit": "m/s",
             "Percent": "unitValue"}
    
   dimensions = ObjectRoot.UnitPreferences
   dimensions.ResetUnits()
   dimensions.SetCurrentUnit("DateFormat", "UTCG")
    
    # Use a for loop to set all the units from the dictionary
   for unit, value in units.items():
       dimensions.SetCurrentUnit(unit, value)

    
   print("Create Satellites...")



    #testlist for trueanomaly
   Truth = TrueAnomalySpacing(TrueAnomalySpace,Nsatellites)
   
   
   ers1 = ObjectRoot.CurrentScenario.Children.New(AgESTKObjectType.eSatellite, "ERS1")
   ers1.SetPropagatorType(AgEVePropagatorType.ePropagatorJ4Perturbation)
   j4 = ers1.Propagator
   interval = j4.EphemerisInterval
   interval.SetExplicitInterval(SimStart, SimEnd)
   j4.Step = 60.00
   oOrb = j4.InitialState.Representation
   oOrb.Epoch = SimStart
       # need to edit true anomaly for the future
   classical = j4.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)       
   classical.CoordinateSystemType = AgECoordinateSystem.eCoordinateSystemJ2000
   classical.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
   trueAnomaly = classical.Location
   trueAnomaly.Value = 0
        

   classical.SizeShapeType = AgEClassicalSizeShape.eSizeShapeSemimajorAxis
   semi = classical.SizeShape
   semi.SemiMajorAxis = Altitude #7163.14
   semi.Eccentricity = 0.0
   classical.Orientation.ArgOfPerigee = 0.0
   classical.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeLAN
   lan = classical.Orientation.AscNode
   lan.Value = 99.38
   classical.Orientation.Inclination = Inclination #98.50
   j4.InitialState.Representation.Assign(classical)

   horizon = ObjectRoot.CurrentScenario.Children["ERS1"].Children.New(AgESTKObjectType.eSensor, "Telescope1")
   horizon.SetPatternType(AgESnPattern.eSnSimpleConic)
   simpleConic  = horizon.Pattern
   simpleConic.ConeAngle = 45
   horizon.SetPointingType(AgESnPointing.eSnPtFixed)
   fixedPt = horizon.Pointing
   azEl = fixedPt.Orientation.ConvertTo(AgEOrientationType.eAzEl)
   azEl.Elevation = -90
   azEl.AboutBoresight = AgEAzElAboutBoresight.eAzElAboutBoresightRotate
   fixedPt.Orientation.Assign(azEl)


        
   ObjectRoot.Rewind()



        #removing the ers1 elevcontours from the 2d window
   ers1.Graphics.ElevContours.IsVisible = False
        
        

  

   j4.Propagate()

   ObjectRoot.Rewind()
    
   
   SatelliteList=[]
   for i in range(int(Nsatellites)-1):
       ers2 = ers1.CopyObject(f"ERS{i}")
       j4 = ers2.Propagator
       classical = j4.InitialState.Representation.ConvertTo(AgEOrbitStateType.eOrbitStateClassical)       
       trueAnomaly = classical.Location
       trueAnomaly.Value = Truth[i+1] #correct true anomaly spacing
       SatelliteList.append(ers2)

   SatelliteList.append(ers1)

   SensorList = []
   for i in SatelliteList:
       SensorList.append(i.Children.Item(0))    
   
   for i in range(len(SensorList)):
        
       SolarExclusionAngle=SensorList[i].AccessConstraints.AddConstraint(AgEAccessConstraints.eCstrLOSSunExclusion)
       SolarExclusionAngle.Angle=90

       LunarExclusionAngle=SensorList[i].AccessConstraints.AddConstraint(AgEAccessConstraints.eCstrLOSLunarExclusion)
       LunarExclusionAngle.Angle=20
      
  # return SensorList



   #star_list_input_data = pd.read_csv(StarList) 
   #star_list_input_data.index = star_list_input_data['name']
   #star_list_input_data['name'] = star_list_input_data['name'].astype(str)

   star_list_input_data = StarList

# create star objects once
   stars = [ObjectRoot.CurrentScenario.Children.New(AgESTKObjectType.eStar, name) for name in star_list_input_data['name']]

# set star properties once
   for star, row in zip(stars, star_list_input_data.itertuples()):
       star.LocationRightAscension = float(row.ra)
       star.LocationDeclination = float(row.dec)
       star.Magnitude = float(row.vmag)
       star.Parallax = float(row.parallax)
       star.ProperMotionDeclination = float(row.propmotdec)
       star.ProperMotionRightAscension = float(row.propmotra)
       star.ProperMotionRadialVelocity = float(row.propmotradialvelocity)

# compute access durations for all stars and all sensors at once
 

   # start_data = []
   # stop_data = []
   # result_data = np.zeros((len(stars), len(SensorList)))

   # for i in range(len(SensorList)):
   #     start_data.append([])
   #     stop_data.append([])
   #     for j in range(len(stars)):
   #         access = SensorList[i].GetAccessToObject(stars[j])
   #         access.ComputeAccess()
   #         ObjectRoot.Rewind()
   #         interval = access.DataProviders["Access Data"]
   #         dimensions.SetCurrentUnit("DateFormat", "UTCG")
   #         result = interval.Exec(SimStart, SimEnd)

   #         result_data[j,i]=np.sum(np.asarray(result.DataSets.ToArray())[:,4].astype(np.int))

   #         dimensions.SetCurrentUnit("DateFormat", "EpSec")

   #         starts = np.asarray(result.DataSets.ToArray())[:,1].astype(np.float)
   #         stops = np.asarray(result.DataSets.ToArray())[:,2].astype(np.float)
           
           
   start_data = [[] for i in range(len(SensorList))]
   stop_data = [[] for i in range(len(SensorList))]
   result_data = np.zeros((len(stars), len(SensorList)))


          


   for i in range(len(SensorList)):

       for j in range(len(stars)):
           access = SensorList[i].GetAccessToObject(stars[j])
           access.ComputeAccess()
           ObjectRoot.Rewind()
           interval = access.DataProviders["Access Data"]
           dimensions.SetCurrentUnit("DateFormat", "UTCG")
           result = interval.Exec(SimStart, SimEnd)

           #result_data[j,i]=np.sum(np.asarray(result.DataSets.ToArray())[:,3].astype(np.int))

           dimensions.SetCurrentUnit("DateFormat", "EpSec")

           starts = np.asarray(result.DataSets.ToArray())[:,1].astype(np.float)
           stops = np.asarray(result.DataSets.ToArray())[:,2].astype(np.float)
           
           

           start_data[i].append(starts)
           stop_data[i].append(stops)          
          
           # compute total coverage for each star
   #result_data = np.sum(result_data,axis=1)
   
   start_data = np.asarray(start_data).T
   stop_data = np.asarray(stop_data).T
   
   start_times = []
   stop_times = []
   for i in range(len(stars)):
       start_times_for_star = np.concatenate(start_data[i])
       start_times.append(start_times_for_star)
       stop_times_for_star = np.concatenate(stop_data[i])
       stop_times.append(stop_times_for_star)
   
   start_times = np.asarray(start_times)
   stop_times = np.asarray(stop_times)
   #now result_data is in the format [TotAccesstoStar1,TotAccesstoStar2]

# convert data to seconds once
  # dimensions.SetCurrentUnit("DateFormat", "EpSec")


   StarListOutput = star_list_input_data
   #StarListOutput['Total Coverage (s)'] = result_data
   
# convert star_list_output to dataframe
   StarListOutput['start times'] = 1
   StarListOutput['stop times'] = 1
   
   #StarListOutput['Total Coverage (Days)'] = StarListOutput['Total Coverage (s)'] / (3600 * 24)
   StarListOutput['start times'] = StarListOutput['start times'].astype('object')
   StarListOutput['stop times'] = StarListOutput['stop times'].astype('object')
   for i in range(len(start_times)):
      StarListOutput['start times'][i] = start_times[i].tolist()
      StarListOutput['stop times'][i] = stop_times[i].tolist()


   

   ObjectRoot.Rewind()

   ObjectRoot.CloseScenario()
   
   #maybe break????
   #return stop_times, start_times
   
   starts = [i for i in StarListOutput['start times']]
   stops = [i for i in StarListOutput['stop times']]
   
   #seismo_data_path=[]
   #for folder_name in StarListOutput.index:
       # create the full path to the folder using the provided path and the current index value
       #folder_path = os.path.join('star_data/', str(folder_name))

       # check if the folder exists in the specified path
       #if os.path.isdir(folder_path):
           #seismo_data_path.append(folder_path)
      # else:
           # handle the case where the folder is not found
          # print(f"Folder {folder_name} not found at {folder_path}")

           
   all_masked_data = []
   for i in range(len(StarListOutput.index)):
       
       all_masked_data.append(mask(starts[i],stops[i],(ts_noise_files[StarListOutput['name'][i]]),MissionDuration)) 
       #please note that the csv is gonna have to be changed
       #to actively seek out the given star

   masked_flux = []
   masked_time = []
   for i in range(len(all_masked_data)):
       masked_flux.append(all_masked_data[i]['flux'])
       masked_time.append(all_masked_data[i]['time'])
       
   
       
   StarListOutput['masked time'] = masked_time
   StarListOutput['masked flux'] = masked_flux
   
   num_rows_before = len(StarListOutput)

# identify rows with empty Series entry in the "masked_flux" column
   empty_flux_rows = StarListOutput[StarListOutput['masked flux'].apply(lambda x: isinstance(x, pd.Series) and x.empty)]

# drop the rows with empty Series entry from the DataFrame
   StarListOutput.drop(empty_flux_rows.index, inplace=True)

# count the number of rows in the DataFrame after dropping the empty flux rows
   num_rows_after = len(StarListOutput)

# calculate the number of rows that were dropped
   num_rows_dropped = num_rows_before - num_rows_after
   
   
   reduced_flux = []
   reduced_time = []
   for i in StarListOutput.index:
       sampler = data_sampler(StarListOutput['masked time'][i],
                              StarListOutput['masked flux'][i])
     
       reduced_flux.append(sampler[0])
       reduced_time.append(sampler[1])
       
   
       
   StarListOutput['reduced time'] = reduced_time
   StarListOutput['reduced flux'] = reduced_flux
   

   incs = []
   stds = []
   for i in StarListOutput.index:
       inc = find_inclination(StarListOutput['reduced time'][i],
                          StarListOutput['reduced flux'][i],
                          lmax)
       incs.append(np.nanmean(inc))
       stds.append(np.nanstd(inc))
       
   StarListOutput['inc'] = incs
   StarListOutput['std'] = stds
   
   error = []
   for i in StarListOutput.index:
       percent_err = percent_error(StarListOutput['inc'][i],
                                   StarListOutput['std'][i],
                                   star_list_input_data['inclination'][i])
       error.append(percent_err)
       #important to note that we see the bias from the paper for low-inc stars
       

       
         
     
   
   

   return np.mean(error),num_rows_dropped #need to have a penalty for how many stars werent observed



n_sat = np.arange(5, 6)
mission_dur = np.arange(1,3)
alt_sat = np.arange(6778, 7278, 100)
#alt_sat = np.arange(6778, 7278, 100)
inc_sat = np.arange(50,100,10 )
#inc_sat = np.arange(97,98,0.11

lookup_table = []
'''
for n in n_sat:
    for dur in mission_dur:
        for alt in alt_sat:
           # for inc in inc_sat:
           x =[n,dur,alt]# [n, dur, alt, inc]
           answer = simulation(x)
           lookup_table.append([n, dur, alt, answer])#lookup_table.append([n, dur, alt, inc, answer])
'''
import gc

with open('simulation_results.txt', 'a') as results_file:
    lookup_table = []
    for n in n_sat:
        for alt in alt_sat:
            for inc in inc_sat:
                x = [n,alt,inc]
                print(x)
                answer = simulation(x, ts_noise_files)
                results_file.write(str([n, answer]) + '\n')
                lookup_table.append([n, answer[0],answer[1]])
                gc.collect()
        
        
from pymoo.core.problem import ElementwiseProblem



def obj1(x):
    #cost model function
    base_cost = 1.0  # relative cost
   # inclination_penalty = 0.01 * abs(x[2] - 28)  # 1% penalty per degree
    #altitude_penalty = 0.0005 * abs(x[1] - 500)  # 0.5% penalty per km
    #altitude_penalty = 0.0005 * abs(7880 - 500) 
    total_cost = base_cost #altitude_penalty#+#inclination_penalty
    #return x[0]*x[1]*total_cost
    
    return x[0]*total_cost
'''               
def obj2(x):
    # check if x is in the lookup table
    for row in lookup_table:
        print(row)
        if np.array_equal(x, row[:3]):
            print(x,row[:3])
        #if np.array_equal(x, row[:4]):
            return row[3] #row[4]


    # if x is not in the lookup table, return a large value
    return 1e6
'''
'''
def obj2(x):
    # check if x is in the lookup table
    for row in lookup_table:
        if np.allclose(x, row[:3], rtol=1.e-1, atol=1.e-1):
            print("Match found in lookup table for x:", x)
            if row[3][1] > 0:
                
                # apply penalty to result if rows were dropped
                penalty_factor = 1 + row[3][1] / 10  # increase penalty by 10% for each row dropped
                return row[3][0] * penalty_factor
            else:
                return row[3][0]

    # if x is not in the lookup table, return a large value
    print("Match not found in lookup table for x:", x)
    return 1e6
'''
def obj2(x):
    # check if x is in the lookup table
    for row in lookup_table:
        if np.allclose(x, row[0], rtol=1.e-1, atol=1.e-1):
            print("Match found in lookup table for x:", x)
            
            return row[3]

    # if x is not in the lookup table, return a large value
    print("Match not found in lookup table for x:", x)
    return 1e6
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=1,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([1]),
                         xu=np.array([20]))
# def __init__(self):
#             super().__init__(n_var=1,
#                              n_obj=2,
#                              n_ieq_constr=3,
#                              xl=np.array([1,1,6778]),
#                              xu=np.array([2,2,6978]))

    def _evaluate(self, x, out, *args, **kwargs):
        
        #cost function
        #f1 = 100 * (x[0]**2 + x[1]**2 + x[2]**2)
        
        #this is the first objective function but we have outsourced it
        f1=obj1(x)
        ''' where:
            x1 = Nsatellites [unitless]
            x2 = MissionDuration [year]
            x3 = AltSat [km]
            x4 = IncSat [degree 0-90]
            '''
        # Data Fidelity function
        ''' where:
            x1 = Nsatellites
            x2 = MissionDuration
            x3 = AltSat
            x4 = IncSat
            '''
        #this is the second objective function but we have outsourced it
        f2 = obj2(x)
        #f2 = (x[0]-1**2 + (x[1]-1)**2 + (x[2]-1)**2

        # g1 = int(round(x[0])) - 2 #integer between 1 and N satellites
        # g2 =  int(round(x[1])) - 3
        
        g1 = x[0] - 20 #integer between 1 and N satellites
        #g2 =  x[1] - 5
        #integer between 1 and 5 years
       # g3 = ((x[2] - 6778) % 100) - 1 #altitude constraints
       # g4 = x[3] - 98 #degree constraints
        #g4 = np.maximum(50- x[2], 0) + np.maximum(x[2] - 90, 0)

        out["F"] = [f1, f2]
        out["G"] = [g1]


problem = MyProblem()


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=32,
    #specifying the sampling step size is super important
    sampling=FloatRandomSampling(), 
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               ('n_gen',500),
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F



#print suboptimal and optimal
for i in lookup_table:
    plt.scatter(i[3],obj1(i),color='black')
plt.scatter(lookup_table[1][3],obj1(lookup_table[1]),color='black',label='Suboptimal Configuration')
plt.scatter(lookup_table[0][3],obj1(lookup_table[0]),color='red',label='Optimal Configuration')


plt.legend()
#plt.title('Working Title')
plt.xlabel('Average Percent Error for Stellar Inclination Retrieval')
plt.ylabel('Relative Mission Cost')
plt.show()

'''sats,years,alt,result -- result from known inclination 
[[1, 1, 6778, 130.5649518088539], -- optimal solution
 [1, 1, 6878, 192.21127071641928],
 [1, 2, 6778, 108.46023887575437],
 [1, 2, 6878, 155.5020703148558],
 [2, 1, 6778, 202.24000392536053],
 [2, 1, 6878, 116.61996724291303],
 [2, 2, 6778, 145.6699898039503],
 [2, 2, 6878, 62.683907563122744]]
'''
''' sats, result
[[1, 230.83045357729387], 
[2, 200.2649372449646],
[2, 200.2649372449646],
[3, 196.33499551399717],
[4, 136.76424257500713],
[5, 150.42672370229567],
[6, 38.854696283637686],
[7, 147.88716151470953],
[8, 200.61227389373568],
[9, 202.90598531503778],
[10, 88.91422004574726],
[11, 159.62810284839028],
[12, 120.56441153714374],
[13, 55.84273904075363],
[14, 127.97407845429865],
[15, 250.80192083001242],
[16, 139.7994955642105],
[17, 77.87727075514695],
[18, 157.74014494974563],
[19, 225.5176578981163],
[20, 280.7585537452779]]
'''
''' sats, result, number of stars that failed
[[1, 6778,50,(82.6688302752247, 4)],
[1, 6778,60,(201.49558560987944, 6)],
[1, 6778,70,(212.8585886326259, 4)],
[1, 6778,80,(207.59405134073396, 4)],
[1, 6778,90,(145.0863028932174, 0)],
[1, 6978,50,(78.14562658639912, 5)],
[1, 6978,60,(167.1160146749213, 4)],
[1, 6978,70,(136.11314517179446, 7)],
[1, 6978,80,(257.7004324676587, 5)],
[1, 6978,90,(224.9337228817769, 0)],
[1, 7078,50,(85.49387894653054, 6)],
[1, 7078,60,(58.19907672010974, 6)],
[1, 7078,70,(49.550412310464566, 4)],
[1, 7078,80,(88.8968105243047, 5)],
[1, 7078,90,(100.24078872882008, 0)],
[1, 7178,50,(248.72571939692392, 6)],
[1, 7178,60,(60.138970297472696, 5)],
[1, 7178,70,(73.66483872371785, 5)],
[1, 7178,80,(142.27949214340083, 5)],
[1, 7178,90,(187.53175804855087, 0)],
[1, 7278,50,(63.220510825699726, 6)],
[1, 7278,60,(100.55668586838273, 6)],
[1, 7278,70,(211.59657380647758, 5)],
[1, 7278,80,(61.450917727414705, 6)],
[1, 7278,90,(185.00386230800498, 0)],
[2, 6778,50,(81.36977893717733, 4)],
[2, 6778,60,(200.12643250107084, 6)],
[2, 6778,70,(220.92579540169, 4)],
[2, 6778,80,(206.92727803221345, 4)],
[2, 6778,90,(278.5231883141291, 0)],
[2, 6878,50,(76.08138394110189, 5)],
[2, 6878,60,(173.7239401867829, 4)],
[2, 6878,70,(134.7520689550687, 7)],
[2, 6878,80,(250.88862000828144, 5)],
[2, 6878,90,(135.83672055390915, 0)],
[2, 6978,50,(85.94689649348686, 6)],
[2, 6978,60,(70.55254319653459, 6)],
[2, 6978,70,(49.45307585451644, 4)],
[2, 6978,80,(91.13607343104525, 5)],
[2, 6978,90,(110.76399854168143, 0)],
[2, 7078,50,(250.24322916906883, 6)],
[2, 7078,60,(59.30647681932692, 5)],
[2, 7078,70,(68.15728595920876, 5)],
[2, 7078,80,(169.48488340551194, 5)],
[2, 7078,90,(284.5996725800835, 0)],
[2, 7178,50,(65.73750898735473, 6)],
[2, 7178,60,(103.81253890085962, 6)],
[2, 7178,70,(211.84029272921708, 5)],
[2, 7178,80,(67.20080703800295, 6)],
[2, 7178,90,(141.10498290050873, 0)],
[3, 6778,50,(82.98749388716699, 4)],
[3, 6778,60,(203.02839960287545, 6)],
[3, 6778,70,(230.81426390324992, 4)],
[3, 6778,80,(207.81783736340427, 4)],
[3, 6778,90,(180.85404783576342, 0)],
[3, 6878,50,(84.56807940023975, 5)],
[3, 6878,60,(170.3897614559708, 4)],
[3, 6878,70,(140.12001156271626, 7)],
[3, 6878,80,(251.65311723482046, 5)],
[3, 6878,90,(84.79257575152512, 0)],
[3, 6978,50,(84.79524711587352, 6)],
[3, 6978,60,(57.130087388260804, 6)],
[3, 6978,70,(48.63303236870333, 4)],
[3, 6978,80,(83.12664491167274, 5)],
[3, 6978,90,(227.08956309908018, 0)],
[3, 7078,50,(249.82286993675055, 6)],
[3, 7078,60,(64.72274419992237, 5)],
[3, 7078,70,(73.90160005684575, 5)],
[3, 7078,80,(144.67525441076646, 5)],
[3, 7078,90,(78.13092236072211, 0)],
[3, 7178,50,(64.10035971742444, 6)],
[3, 7178,60,(103.95314251638504, 6)],
[3, 7178,70,(204.95554614969333, 5)],
[3, 7178,80,(68.95619439290738, 6)],
[3, 7178,90,(65.59803753346286, 0)],
[4, 6778,50,(81.88565571035878, 4)],
[4, 6778,60,(202.40019510038297, 6)],
[4, 6778,70,(229.56439190111956, 4)],
[4, 6778,80,(207.43391779719983, 4)],
[4, 6778,90,(178.9343684983107, 0)],
[4, 6878,50,(91.34630605954109, 5)],
[4, 6878,60,(167.05667691928267, 4)],
[4, 6878,70,(129.51848178084782, 7)],
[4, 6878,80,(250.99480214768406, 5)],
[4, 6878,90,(147.75134566901744, 0)],
[4, 6978,50,(85.73239847964813, 6)],
[4, 6978,60,(61.4020249920748, 6)],
[4, 6978,70,(49.57367780309195, 4)],
[4, 6978,80,(95.60314054760144, 5)],
[4, 6978,90,(219.39267602344924, 0)],
[4, 7078,50,(249.46300999319942, 6)],
[4, 7078,60,(60.783292820173926, 5)],
[4, 7078,70,(67.20624250315649, 5)],
[4, 7078,80,(177.76565316011852, 5)],
[4, 7078,90,(112.76589128863672, 0)],
[4, 7178,50,(64.59630335828068, 6)],
[4, 7178,60,(104.60601935827016, 6)],
[4, 7178,70,(215.28759701754382, 5)],
[4, 7178,80,(68.65242442095544, 6)],
[4, 7178,90,(131.53139222922312, 0)]]

[5, (81.47216451391469, 4)],
[5, (203.10839321044415, 6)],
[5, (229.4652915220546, 4)],
[5, (206.90930184810668, 4)],
[5, (206.4430820393768, 0)],
[5, (80.49534258590603, 5)],
[5, (160.9712635441016, 4)],
[5, (144.94565601611632, 7)],
[5, (251.63489042073053, 5)],
'''

import matplotlib.pyplot as plt

# Plot the pareto front


# Plot the results from the lookup table
possible_soln = []
for row in lookup_table:
    x = row[:2]
    f1 = obj1(x)
    f2 = obj2(x)
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]
    
plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x')#label='Possible Solutions')
plt.scatter(F[:,0], F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal')
plt.xlabel('Relative Cost')
plt.ylabel('Percent Error of Inclination Retrieval')
plt.title('NSGA-II  Results for Cost and Inclination Retrieval Objectives')
plt.legend()
plt.ylim(0,300)
plt.show()

#another plot that plots the original error assessnment against number of sats
possible_soln = []
for row in lookup_table:
    x = row[:3]
    f1 = row[0]
    f2 = row[3]
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]

plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x',label='Possible Solutions')
plt.xticks(np.arange(0,21,1)) 
plt.ylabel('Percent Error of Inclination Retrieval') 
plt.xlabel('Number of Satellites') 

plt.scatter(np.round(X[:,0]), F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal')
plt.legend()
plt.show()

#another plot that shows the obj2 error assessment vs number of sats
possible_soln = []
for row in lookup_table:
    x = row[:3]
    f1 = row[0]
    f2 = obj2(x)
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]

plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x',label='Possible Solutions')
plt.xticks(np.arange(0,20,1)) 
plt.ylabel('Percent Error of Inclination Retrieval') 
plt.xlabel('Number of Satellites') 

plt.scatter(np.round(X[:,0]), F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal')
plt.legend()
plt.show()


#another plot that shows the obj2 error assessment vs inc
possible_soln = []
for row in lookup_table:
    x = row[:3]
    f1 = row[2]
    f2 = obj2(x)
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]

plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x',label='Possible Solutions')
plt.xticks(np.arange(50,90,10)) 
plt.ylabel('Percent Error of Inclination Retrieval') 
plt.xlabel('i$^{\circ}$') 

plt.scatter(np.round(X[:,2]), F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal')
plt.legend()
plt.show()

#another plot that shows the obj2 error assessment vs alt
possible_soln = []
for row in lookup_table:
    x = row[:3]
    f1 = row[2]
    f2 = obj2(x)
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]

plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x',label='Possible Solutions')
plt.xticks(np.arange(6678,7280,100)) 
plt.ylabel('Percent Error of Inclination Retrieval') 
plt.xlabel('Altitude [km]') 
plt.scatter(np.round(X[:,2]), F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal')
plt.legend()
plt.show()