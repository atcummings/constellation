# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 08:40:18 2023

@author: andre
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import astropy.stats
import emcee
import scipy.optimize

import math
import geopandas as gpd   
import pyproj
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

gdf = gpd.read_file('gb_10km.shp')   

# define input and output CRSs using the Proj4 string
input_crs = pyproj.CRS.from_proj4(
    '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs')
#might need to reattach this string into one long thing
output_crs = pyproj.CRS.from_epsg(4326)

centroid_points = []

# loop through each Polygon geometry in the GeoDataFrame
for geometry in gdf.geometry:
    # create a shapely Polygon object from the geometry
    polygon = Polygon(geometry)
    # get the centroid point of the polygon
    centroid = polygon.centroid
    # add the centroid point to the list
    centroid_points.append(centroid)

# create a new GeoDataFrame from the centroid points
centroid_gdf = gpd.GeoDataFrame(geometry=centroid_points, crs=gdf.crs)

# create transformer object
transformer = pyproj.Transformer.from_crs(input_crs, output_crs)
centroid_gdf['lat'], centroid_gdf['lon'] = transformer.transform(centroid_gdf.geometry.x.values, centroid_gdf.geometry.y.values)

# apply transformer to get longitude and latitude coordinates


data = np.asarray([centroid_gdf['lon'],centroid_gdf['lat']]).T

# generate sample data


# compute Delaunay triangulation
tri = Delaunay(data)

# compute Alpha Shape parameter
alpha = 0.5 * np.sqrt(np.mean(np.sum((data[tri.simplices[:, 0]] - data[tri.simplices[:, 1]])**2, axis=1)))

# get indices of boundary triangles
simplices = tri.simplices[tri.neighbors[:, 0] == -1]

# get boundary vertices
vertices = np.unique(simplices)

# compute Delaunay triangulation of boundary vertices
tri_boundary = Delaunay(data[vertices])

# get indices of triangles in the boundary Delaunay triangulation
simplices_boundary = tri_boundary.simplices

# get boundary edges
edges = set()
for simplex in simplices_boundary:
    for i in range(3):
        for j in range(i+1, 3):
            edge = (vertices[simplex[i]], vertices[simplex[j]])
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            edges.add(edge)

# convert edges to numpy array
border = np.array(list(edges))

border_lon=data[border[:,0],0]
border_lat = data[border[:,0],1]
#downsampled to hopefully preserve shape
down_lon = border_lon[::5]
down_lat=border_lat[::5]

down_data = np.asarray([down_lat,down_lon]).T

boundary = (np.round(down_data,decimals=5)).tolist() #meter precision


def clockwise_sort(coords):
    # Step 1: Calculate the centroid
    latitudes = [c[0] for c in coords]
    longitudes = [c[1] for c in coords]
    centroid_lat = sum(latitudes) / len(coords)
    centroid_lon = sum(longitudes) / len(coords)

    # Step 2: Convert to polar coordinates
    polar_coords = []
    for coord in coords:
        dx = coord[1] - centroid_lon
        dy = coord[0] - centroid_lat
        r = math.sqrt(dx*dx + dy*dy)
        theta = math.atan2(dy, dx)
        polar_coords.append((theta, r, coord))

    # Step 3: Sort by polar angle
    sorted_polar_coords = sorted(polar_coords)

    # Step 4: Convert back to Cartesian coordinates
    sorted_coords = [(c[2][0], c[2][1]) for c in sorted_polar_coords]

    return sorted_coords

boundary = clockwise_sort(boundary)
boundary = [[x for x in tup] for tup in boundary]




# Define the rectangle dimensions and distance
width = 4224    # meters
height = 3380   # meters
distance = 450000   # meters

# Calculate the horizontal half angle
horizontal_half_angle = math.atan(width / (2 * distance)) * (180 / math.pi)

# Calculate the vertical half angle
vertical_half_angle = math.atan(height / (2 * distance)) * (180 / math.pi)

# Print the results



import os
import platform
import time

from agi.stk12.stkobjects import *
from agi.stk12.stkutil import *
from agi.stk12.utilities.colors import *

from agi.stk12.stkobjects import (
    AgEClassicalLocation,
    AgEClassicalSizeShape,
    AgECvBounds,
    AgECvResolution,
    AgEFmCompute,
    AgEFmDefinitionType,
    AgEOrientationAscNode,
    AgESTKObjectType,
    AgEVePropagatorType,
)


from agi.stk12.stkutil import AgEOrbitStateType

startTime = time.time()

"""
SET TO TRUE TO USE ENGINE, FALSE TO USE GUI
"""
if platform.system() == "Linux":
    # Only STK Engine is available on Linux
    useStkEngine = True
else:
    # Change to true to run engine on Windows
    useStkEngine = False

if useStkEngine:
    from agi.stk12.stkengine import STKEngine

    print("Launching STK Engine...")
    stk = STKEngine.StartApplication(noGraphics=True)


    stkRoot = stk.NewObjectRoot()

else:
    from agi.stk12.stkdesktop import STKDesktop

 
    print("Launching STK...")
    stk = STKDesktop.StartApplication(visible=True, userControl=True)

    stkRoot = stk.Root


stkRoot.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")

# Create new scenario
print("Creating scenario...")
stkRoot.NewScenario("PythonEngineExample")
scenario = stkRoot.CurrentScenario

areaTarget = scenario.Children.New(2, 'MyAreaTarget') # eAreaTarget
areaTarget.AreaType = 1 # ePattern
patterns = areaTarget.AreaTypeData
areaTarget.CommonTasks.SetAreaTypePattern(boundary)

areaTarget.AutoCentroid = True


file_path = 'eo_simulation.txt'


def simulation(x,file):

    n_sats = x[0]
    MissionDuration = x[1]
    altitude = x[2]
    inclination = x[3]
    n_planes = x[4]

# Set time period
    SimStart = "1 Jul 2002 00:00:00.00"
    start_time = datetime.strptime(SimStart, "%d %b %Y %H:%M:%S.%f")
    time_delta = MissionDuration*timedelta(days=365)
    duration = start_time+time_delta
    #this is a new way to specify the time duration of the entire simulation
    SimEnd = duration.strftime("%d %b %Y %H:%M:%S.%f")
    
    scenario.SetTimePeriod(SimStart,SimEnd)
    if not useStkEngine:
    # Graphics calls are not available when running STK Engine in NoGraphics mode
        stkRoot.Rewind()







# Create the constellation 
    constellation = scenario.Children.New(
        AgESTKObjectType.eConstellation, "SatConstellation"
        )

# Insert the constellation of Satellites
    numOrbitPlanes = n_planes
    numSatsPerPlane = n_sats

    stkRoot.BeginUpdate()
    for orbitPlaneNum, RAAN in enumerate(
            range(0, 180, 180 // numOrbitPlanes), 1):  # RAAN in degrees

        for satNum, trueAnomaly in enumerate(
                range(0, 360, 360 // numSatsPerPlane), 1):  # trueAnomaly in degrees

                satellite = scenario.Children.New(
                AgESTKObjectType.eSatellite, f"Sat{orbitPlaneNum}{satNum}"
            )

      
                satellite.SetPropagatorType(AgEVePropagatorType.ePropagatorTwoBody)

       
                twoBodyPropagator = satellite.Propagator
                keplarian = twoBodyPropagator.InitialState.Representation.ConvertTo(
                    AgEOrbitStateType.eOrbitStateClassical.eOrbitStateClassical
            )

                keplarian.SizeShapeType = AgEClassicalSizeShape.eSizeShapeSemimajorAxis
                keplarian.SizeShape.SemiMajorAxis = altitude  # km
                keplarian.SizeShape.Eccentricity = 0

                keplarian.Orientation.Inclination = inclination  # degrees
                keplarian.Orientation.ArgOfPerigee = 0  # degrees
                keplarian.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeRAAN
                keplarian.Orientation.AscNode.Value = RAAN  # degrees

                keplarian.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
                keplarian.Location.Value = trueAnomaly + (360 // numSatsPerPlane / 2) * (
                    orbitPlaneNum % 2
            )  

        
                satellite.Propagator.InitialState.Representation.Assign(keplarian)
                satellite.Propagator.Propagate()
        
                sensor = satellite.Children.New(20,'Sensor') # eSensor
            
                sensor.SetPatternType(AgESnPattern.eSnRectangular)
                rectangle  = sensor.Pattern
                rectangle.HorizontalHalfAngle=horizontal_half_angle 
        
                rectangle.VerticalHalfAngle=vertical_half_angle 
        
                sensor.SetPointingType(AgESnPointing.eSnPtFixed)
                fixedPt = sensor.Pointing
                azEl = fixedPt.Orientation.ConvertTo(AgEOrientationType.eAzEl)
                azEl.Elevation = 90
                azEl.AboutBoresight = AgEAzElAboutBoresight.eAzElAboutBoresightRotate
                fixedPt.Orientation.Assign(azEl)

        
     
                constellation.Objects.AddObject(satellite)

    stkRoot.EndUpdate()
# Create chain



####### 
# Coverage
#######
    coverageDefinition = scenario.Children.New(
        AgESTKObjectType.eCoverageDefinition, "CoverageDefinition"
        )

# Set grid bounds type
    grid = coverageDefinition.Grid
    grid.BoundsType = AgECvBounds.eBoundsCustomRegions

# Add US shapefile to bounds
    covGrid = coverageDefinition.Grid
    bounds = covGrid.Bounds

    bounds = coverageDefinition.Grid.Bounds
    bounds.AreaTargets.Add('AreaTarget/MyAreaTarget')
#Define the Grid Resolution
    Res = covGrid.Resolution
    Res.LatLon = .5   #deg
#Set the sensor as the Asset

#get a list of the string ids for every sensor. Note, this will fail 
#with multiple area targets
    sensor_list =constellation.Objects.AvailableObjects[2::2]

    for sensor in sensor_list:
        coverageDefinition.AssetList.Add(sensor)
        
    
    coverageDefinition.ComputeAccesses()




# Create figure of merit
    figureOfMerit = coverageDefinition.Children.New(
        AgESTKObjectType.eFigureOfMerit, "FigureOfMerit"
        )

# Set the definition and compute type
    figureOfMerit.SetDefinitionType(AgEFmDefinitionType.eFmRevisitTime)
    definition = figureOfMerit.Definition
#definition.Satisfaction.EnableSatisfaction=True
#definition.Satisfaction.SatisfactionThreshold = 60
    definition.SetComputeType(AgEFmCompute.eAverage)

    fomDataProvider = figureOfMerit.DataProviders.GetDataPrvFixedFromPath("Overall Value")
    fomResults = fomDataProvider.Exec()
    try: 
        minRevisit = fomResults.DataSets.GetDataSetByName("Minimum").GetValues()[0]
        maxRevisit = fomResults.DataSets.GetDataSetByName("Maximum").GetValues()[0]
        avgRevisit = fomResults.DataSets.GetDataSetByName("Average").GetValues()[0]
    except STKRuntimeError:
        days = 0
        
    
# Computation time
    for obj in scenario.Children:
        if obj.InstanceName != 'MyAreaTarget':
            obj.Unload()
    
    
    days = avgRevisit / (24*3600)
    file.write(str([x,days])+ '\n')
    return days


n_sats = np.arange(1,11)
duration = np.arange(1,2)
altitude = np.arange(6821,6921,100) #np.arange(6770,7270,100)
inclination = np.arange(60,65,5)
n_planes = np.arange(1,2)

'''
n_sats = np.arange(1,3)
duration = np.arange(1,2)
altitude = np.arange(6821,6921,100) #np.arange(6770,7270,100)
inclination = np.arange(64,65,1)
n_planes = np.arange(1,5)
'''
import gc

with open(file_path, 'a') as results_file:
    lookup_table = []
    for n in n_sats:
        for dur in duration:
            for alt in altitude:
                for inc in inclination:
                    for planes in n_planes:
                        x = [n,dur,alt,inc,planes]
                        print(x)
                        answer = simulation(x,results_file)
                        lookup_table.append([x, answer])
                        gc.collect()
                        
        
        
        
############################
## OPTIMIZATION CODE      ##
############################        
from pymoo.core.problem import ElementwiseProblem



def obj1(x):
    #cost model function
    base_cost = 1.0  # relative cost
    inclination_penalty = 0.01 * abs(x[3] - 28)  # 1% penalty per degree
    n_planes_penalty = 0.05 * abs(x[4] - 1) # 5% penalty per extra plane
    altitude_penalty = 0.0005 * abs(x[2] - 6821)  # 0.5% penalty per km
    #altitude_penalty = 0.0005 * abs(7880 - 500) 
    if n_planes_penalty > 1:
        total_cost = (base_cost +inclination_penalty+
                      n_planes_penalty+altitude_penalty)*x[0]*x[4]
    else: 
        total_cost = (base_cost +inclination_penalty+
                      n_planes_penalty+altitude_penalty)*x[0]
    #return x[0]*x[1]*total_cost
    
    return total_cost

def obj2(x):
    # check if x is in the lookup table
    for row in lookup_table:
        if np.allclose(x, row[0], rtol=1.e-1, atol=1.e-1):
            print("Match found in lookup table for x:", x)
            
            return row[1]

    # if x is not in the lookup table, return a large value
    print("Match not found in lookup table for x:", x)
    return 1e6

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=5,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([1,1,6770,50,1]),
                         xu=np.array([5,1,7200,90,4]))
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
        
        g1 = int(round(x[0])) - 5 #integer between 1 and N satellites
        #g2 =  x[1] -) 5
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


'''
# Print data to console
print("\nThe minimum coverage duration is {a:4.2f} min.".format(a=minAccessDuration))
print("The maximum coverage duration is {a:4.2f} min.".format(a=maxAccessDuration))
print("The average coverage duration is {a:4.2f} min.".format(a=avgAccessDuration))
print(
    "--- Coverage computation: {a:0.3f} sec\t\tTotal time: {b:0.3f} sec ---".format(
        a=sectionTime, b=totalTime
    )
)
'''
# stkRoot.CloseScenario()
# stk.ShutDown()



# Plot the pareto front
# Plot the results from the lookup table
possible_soln = []
for row in lookup_table:
    x = row[0]
    f1 = obj1(x)
    f2 = obj2(x)
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]
    
plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x',label='Possible Solutions')
plt.scatter(F[:,0], F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal Solutions')
plt.plot([0,max(possible_soln[0])],[360,360],'--',color = 'gray',label='Simulation Time Boundary')
plt.xlabel('Relative Cost')
plt.ylabel('Revisit Time [Days]')
plt.title('NSGA-II  Results for Cost and Revisit Time Objectives')
plt.legend(bbox_to_anchor=(0.5,-0.15))
#plt.ylim(0,300)
plt.show()

#another plot that plots the revisit time against number of sats

possible_soln = []
for row in lookup_table:
    f1 = row[0][0] * row[0][4]
    f2 = row[1]
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]

plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x',label='Possible Solutions')
plt.xticks(np.arange(0,max(possible_soln[0])+1,1)) 
plt.ylabel('Revisit Time [Days]') 
plt.xlabel('Number of Satellites') 
plt.plot([1,max(possible_soln[0])],[360,360],'--',color = 'gray',label='Simulation Time Boundary')

plt.scatter(np.round(X[:,0]), F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal')
plt.legend(bbox_to_anchor=(0.5,-0.15))
plt.show()


#inc vs revisit time
possible_soln = []
for row in lookup_table:
    x = row[0]
    f1 = row[0][3]
    f2 = obj2(x)
    possible_soln.append([f1,f2])
possible_soln = np.asarray(possible_soln).T#row[3][0]

plt.scatter(possible_soln[0],possible_soln[1], c='r', marker='x',label='Possible Solutions')
#plt.xticks(np.arange(50,90,10)) 
plt.ylabel('Revisit Time [Days]') 
plt.xlabel('Constellation Orbital Inclination [i$^{\circ}$]') 
plt.plot([0,max(possible_soln[0])],[360,360],'--',color = 'gray',label='Simulation Time Boundary')
z1 = [48,48]
z2 = [0,360]
plt.plot(z1,z2,linestyle='dotted',color='gray',label='Minimum Latitude of Coverage Area')
plt.scatter(np.round(X[:,3]), F[:,1],s=50 ,c='blue',marker='D',label='Pareto Optimal')
plt.legend(bbox_to_anchor=(0.5,-0.15))

plt.show()
