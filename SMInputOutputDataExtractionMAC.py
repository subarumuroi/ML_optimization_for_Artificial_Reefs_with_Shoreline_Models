# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:34:19 2023

@author: s5230048
"""

"""

Script to extract data from Mike SM (for Subaru)

-----------------------
Instructions
-----------------------
1. Install required dependencies: 
   - mikeio, matplotlib, datetime
2. Specify the directory paths in the "user inputs section"
3. Ensure all sbw configuration data is in the ModelConfig.csv file
   - Note that the entry in the "Model Run" row MUST be the same as the .mfm file. If the model file is c7.mfm, then enter c7 as the Model Run   

-----------------------
Assumptions:
-----------------------
 - Mike variable Coastline Y is the longshore coordinate and Coastline X is the shoreline position 
   ie assumes model domain is oriented with alongshore in the y-direction and crossshore in the x-direction
 - the shoreline .dfs1 file has the same name for each of the model runs
 - each model has the same discretisation (ie same number of shore normal profiles in the same longshore locations)

----------------------
Version History
----------------------
Version 01, nc, July 5, 2023
 - outputs a single .csv file for each model run containing all profiles 

Version 02, nc, July 10, 2023
 - more efficient creation of dataframe from dataset
 - extracts input parameters from .mfm file
 - time step selection and outputs another .csv file of concatenated profiles from all model runs at a particular time step
 - add basic shoreline position stats output to .csv file (max, min, mean)
 - added ModelConfig.csv file as a way to read in sbw configuration data 
 
Version 03, nc, August 2, 2023
 - SBW configuration (distance, length, width) extracted from .mesh file name convenstion, e.g. a 150 100 50.mesh 
 

"""

#%% user inputs

# root directory containing all "XXXX.mfm - ResultFiles" result folders
RootDirectory = r"\Volumes\Extreme SSD\PhD\Models" 

# root directory containing all "XXXX.mesh" files used to obtain distance, length and width data from file name
MeshFileDirectory = r'\Volumes\Extreme SSD\PhD\Input files'

# output directory for .csv file
global OutputDirectory
OutputDirectory = r"\Volumes\Extreme SSD\PhD\Extracted (csv) Data"

# Name of the file with the shoreline data
ShorelineFilename_dfs1 = 'coastline.dfs1' # name of result file with shoreline data in it *** assumed to be the same for all runs
TimeofDataExtraction = 365 # time in DAYS after t0 to extract the SL profile data

#%% packages
import os
import mikeio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

#%%
def splitLettersNumbers(s):
    letter = s.rstrip('0123456789')
    number = s[len(letter):]
    return letter, number        


#%%

def getInputParameters(mfm_file_path):
    
    '''
    Get input model parameters from model .mfm control file
    '''
    
    # Initialise output dict
    ModelRun = mfm_file_path.split('\\')[-1].split('.')[0]
    InputData = {'Model Run' : ModelRun}
    
    # split run id into letter and number - used to match with .mesh filename extarction of sbw distance, length and width
    Run_Letter, Run_Number = splitLettersNumbers(ModelRun)
    InputData['Run Letter'] = Run_Letter
    InputData['Run Number'] = Run_Number
    
    # read .mfm file and extract wave boundary condition data
    # section tags in .mfm are:
    # [SPECTRAL_WAVE_MODULE]
    #   [BOUNDARY_CONDITIONS]
    #       [CODE_3]
    #           constant_values = 2.5, 7.0, 180.0, 1.0, 0.1, 16.0, 270.0, 32.0
    
    f = open(mfm_file_path, 'r')
    while True:
        nstr = f.readline()
        
        # wave module parameters ----------------------------------------------
        if '[SPECTRAL_WAVE_MODULE]' in nstr:
            while True:
                nstr = f.readline()
                if 'EndSect  // SPECTRAL_WAVE_MODULE' in nstr:
                    break
                if '[BOUNDARY_CONDITIONS]' in nstr:
                    while True:
                        nstr = f.readline()
                        if 'EndSect  // BOUNDARY_CONDITIONS' in nstr:
                            break
                        if '[CODE_3]' in nstr:
                            while True:
                                nstr = f.readline()
                                # print('[CODE_3]')
                                if 'EndSect  // CODE_3' in nstr:
                                    break
                                if 'constant_values' in nstr: # we are at the wave boundary condition line
                                    tmp = nstr.split('=')[-1].split(',')
                                    InputData['Wave Height (m)'] = float(tmp[0])
                                    InputData['Wave Period (s)'] = float(tmp[1])
                                    InputData['Wave Direction (deg)'] = float(tmp[2])

        # test for end of file
        if len(nstr) == 0:
            break
        
    # close the file
    f.close()
    
    return InputData

#%%

def getShorelineData(file_path):
    '''
    Get shoreline data from model .dfs1 result file
    '''
    # get the data
    ds = mikeio.read(file_path) # load .dsf1 file into (xarray) dataset
    
    # convert dataset to dataframe
    df = pd.DataFrame(index=ds['Coastline Y'].isel(time=0).values.T, data=ds['Coastline X'].values.T)
    df.columns = ds.time.strftime("%Y-%m-%d %H:%M:%S").tolist()
    df.index.name = 'Longshore Position (Coastline Y) (m)'

    return df

#%%

def getShorelineData_AtTime(file_path, TimeofDataExtraction):
    '''
    Get shoreline data from model .dfs1 result file
    '''
    # get the data
    ds = mikeio.read(file_path) # load .dsf1 file into (xarray) dataset
    
    # get index of requested time step
    ExtractionTime = ds.time[0] + datetime.timedelta(days=TimeofDataExtraction)

    dt = ExtractionTime - ds.time[-1]
    if dt.total_seconds() > 0: # then requested time is after the end of the results file
        print('')
        print('    *** ERROR ***')
        print('    Requested time step is after end of model results!')
        print('')
        check='ERROR'
        return check
    extractionID = ds.time.get_indexer([ExtractionTime], method='nearest')[0]

    # convert dataset to dataframe
    df = pd.DataFrame(index=ds['Coastline Y'].isel(time=0).values.T)
    df[ds.time[extractionID].strftime("%Y-%m-%d %H:%M:%S")] = ds['Coastline X'].isel(time=extractionID).values.T
    df.index.name = 'Longshore Position (Coastline Y) (m)'

    check='OK'
    
    return df, check

#%%

def getShorelineStats(df):
    
    #% get basic stats
    df_stats = pd.DataFrame() 
    df_stats['Shoreline Max (m)'] = df.max(axis=0)
    df_stats['Shoreline Max Alongshore Location (m)'] = df.idxmax(axis=0)
    df_stats['Shoreline Min (m)'] = df.min(axis=0)
    df_stats['Shoreline Min  Alongshore Location (m)'] = df.idxmin(axis=0)
    df_stats['Shoreline Mean (m)'] = df.mean(axis=0)
    df_stats = df_stats.T
    
    return df_stats
    
#%%

def writeCSVfile(InputData, df, df_stats):

    # write csv file
    csvFileName = InputData['Model Run'] + '.csv'
    with open(OutputDirectory + '\\' + csvFileName,'w') as f:
        # write file header lines
        f.write('Mike Shoreline Morphology Results\n')
        f.write('\n')
        f.write('Model Run, ' + InputData['Model Run'] + '\n')
        f.write('\n')
        f.write('Wave height (m), ' + str(InputData['Wave Height (m)']) + '\n')
        f.write('Wave period (s), ' + str(InputData['Wave Period (s)']) + '\n')
        f.write('Wave direction (deg), ' + str(InputData['Wave Direction (deg)']) + '\n')
        f.write('\n')
        f.write('SBW distance offshore (m), ' + str(InputData['SBW distance offshore (m)']) + '\n')
        f.write('SBW length (m), ' + str(InputData['SBW length (m)']) + '\n')
        f.write('SBW width (m), ' + str(InputData['SBW width (m)']) + '\n')
        f.write('\n')
        f.write('Variable, Shoreline Position (Coastline X) (m) \n')
        f.write('Source File, ' + InputData['Shoreline Data File'] + '\n')
        f.write('\n')

        # add stats
        f.write('Statistics \n')
        df_stats.to_csv(f, lineterminator='\n')
        f.write('\n')
        
        # f.write(df.columns.values[1:] + '\n')
        f.write('Profiles \n')
        df.to_csv(f, lineterminator='\n')

    return

#%%

def makeProfilePlot(InputData, ds):

    for ii in range(len(ds.time)):
        da = ds['Coastline X'].isel(time=ii) # create data array for current time step
        if ii == 0: 
            ax = da.plot(title='')
        else:
            da.plot(ax=ax, title=str(ds.time[0]) + ' to ' + str(ds.time[-1]))
    ax.grid(True)
    plt.savefig(InputData['Shoreline Data File'] + '.png', dpi=300.)

    return

#%%

def getSBWconfigFromMeshFile(MeshFileDirectory):
    
    # initialise lists
    code = []
    distance = []
    length = []
    width = []
    
    # parse sbw config data from .mesh filename
    for FileFolder in os.listdir(MeshFileDirectory):
        path = os.path.join(RootDirectory, FileFolder)
        if path.endswith(".mesh"):
            tmp = path.split('\\')[-1].split('.')[0].split(' ')
            code.append(tmp[0])
            distance.append(tmp[1])
            length.append(tmp[2])
            width.append(tmp[3])
            
    # create output dataframe
    output = {'Code' : code,
              'Distance' : distance,
              'Length' : length,
              'Width' : width}
    df = pd.DataFrame(data=output)

    return df
    
#%% Create separate .csv files for each results file

# Get list of model folders
ListOfModelFolders = [];
for ModelFolder in os.listdir(RootDirectory):
    ListOfModelFolders.append(os.path.join(RootDirectory, ModelFolder))

# create output directory for processed shoreline data
if not os.path.exists(OutputDirectory):
    os.makedirs(OutputDirectory)

# get SBW config fdata (distance, length, width)
df_ModelConfig = getSBWconfigFromMeshFile(MeshFileDirectory)

import re
match = re.match(r"([a-z]+)([0-9]+)", 'foofo21', re.I)
if match:
    items = match.groups()
    
# initia;lise processing log file
logFile = open(OutputDirectory + '\logFile.txt','w')

# loop over model folders
df_SingleTime_Out = pd.DataFrame()   

ii = 0
for FileFolder in os.listdir(RootDirectory):
    path = os.path.join(RootDirectory, FileFolder) 

    #%% get the input parameters
    if path.endswith(".mfm"): # we have a model control file
        ii = ii + 1
        #if ii > 9:
        #    sys.exit()
        print(' ')
        print('  Processing ' + FileFolder + ' ...')
        print('    Extracting input data from .mfm file ...')
        InputData = getInputParameters(path)

        #%% get reef config data and add to df
        
        # find index of df_ModelConfig row containing current sbw confis (Run Letter)
        idx = df_ModelConfig.index[df_ModelConfig['Code']==InputData['Run Letter']]
        InputData['SBW distance offshore (m)'] = float(df_ModelConfig['Distance'].iloc[idx].values[0])
        InputData['SBW length (m)'] = float(df_ModelConfig['Length'].iloc[idx].values[0])
        InputData['SBW width (m)'] = float(df_ModelConfig['Width'].iloc[idx].values[0])
            
        #%% get and process shoreline data
        ResultPath = RootDirectory + '\\' + FileFolder + " - Result Files"
        ResultFilePath = os.path.join(ResultPath,ShorelineFilename_dfs1)
        print('    Extracting ALL shoreline data from .dfs1 file and writing to .csv ...')
        InputData['Shoreline Data File'] = ResultFilePath
        
        #%% get ALL shoreline data --------------------------------------
        df = getShorelineData(ResultFilePath) 
        # get stats
        df_stats = getShorelineStats(df)
        # write to .csv
        writeCSVfile(InputData, df, df_stats)
        
        #%% get shoreline data at specified time interval ----------------
        print('    Extracting shoreline data for specified time from .dfs1 file...')
        df, check = getShorelineData_AtTime(ResultFilePath, TimeofDataExtraction)
        InputData['Extraction Duration (days)'] = TimeofDataExtraction
        InputData['Extraction Timestamp'] = df.columns[0]
        df.columns = [FileFolder]
        if check == 'ERROR':  # user specified extraction tim echeck
            continue

        # get stats
        df_stats = getShorelineStats(df)       
        df_stats = pd.concat([df_stats, pd.DataFrame(index=['Alongshore Position (m)'], data=['SL (m)'], columns=[FileFolder])])
        df_stats.columns = [FileFolder]
        df_stats = df_stats.reset_index()
        df['index'] = df.index
        df = df.reset_index(drop=True)
        df = df[['index',FileFolder]]

        # convert InputData dict to a dataframe ready for concatenation        
        items = InputData.items()
        df_InputParameters = pd.DataFrame({'index': [i[0] for i in items], FileFolder: [i[1] for i in items]})
        
        # concatenate results for current model run
        df_Out_tmp = pd.concat([df_InputParameters, df_stats, df])
        df_Out_tmp = df_Out_tmp.set_index('index', drop=True)

        # concatenate results for current model run with previous model runs       
        df_SingleTime_Out = pd.concat([df_SingleTime_Out, df_Out_tmp], axis=1)

        # update log
        logFile.write('Processing '+FileFolder[0:-4]+ ' - OK\n')
        
logFile.close()

#%%
print(' ')
print('  Writing all profiles at single time')
csvFileName = 'AllRuns_SingleTime.csv'
df_SingleTime_Out.to_csv(OutputDirectory + '\\' + csvFileName, header=False)

                
#%% end