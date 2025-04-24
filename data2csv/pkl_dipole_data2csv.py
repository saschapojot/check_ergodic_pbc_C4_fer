import numpy as np
from datetime import datetime
import sys
import re
import glob
import os
import json
from pathlib import Path
import pandas as pd
import pickle
#this script extracts effective data from pkl files
# for dipole

if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
TStr=sys.argv[2]
init_path=sys.argv[3]
summary_obs_name="dipole"
dataRoot=f"./dataAll/N{N}/T{TStr}/init_path{init_path}/"
csv_out_path=f"./dataAll/N{N}/T{TStr}/"


def parseSummary(summary_obs_name):
    startingFileInd=-1

    lag=-1
    sweep_to_write=-1
    smrFile=dataRoot+"/summary_"+summary_obs_name+".txt"

    summaryFileExists=os.path.isfile(smrFile)
    if summaryFileExists==False:
        return startingFileInd,-1

    with open(smrFile,"r") as fptr:
        lines=fptr.readlines()

    for oneLine in lines:
        #match startingFileInd
        matchStartingFileInd=re.search(r"startingFileInd=(\d+)",oneLine)

        if matchStartingFileInd:
            startingFileInd=int(matchStartingFileInd.group(1))
        startingFileInd=35
        #match lag
        matchLag=re.search(r"lag=(\d+)",oneLine)
        if matchLag:
            lag=int(matchLag.group(1))

        #match sweep_to_write
        match_sweep_to_write=re.search(r"sweep_to_write=(\d+)",oneLine)

        if match_sweep_to_write:
            sweep_to_write=int(match_sweep_to_write.group(1))

    return startingFileInd,lag,sweep_to_write


def sort_data_files_by_flushEnd(summary_obs_name,varName):
    dataFolderName=dataRoot+"/U_dipole_dataFiles/"+varName+"/"

    dataFilesAll=[]
    flushEndAll=[]
    for oneDataFile in glob.glob(dataFolderName+"/flushEnd*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"flushEnd(\d+)",oneDataFile)
        if matchEnd:
            flushEndAll.append(int(matchEnd.group(1)))

    endInds=np.argsort(flushEndAll)
    sortedDataFiles=[dataFilesAll[i] for i in endInds]

    return sortedDataFiles


def one_dipole_component_extract_ForOneT(startingFileInd,lag,component_name,sweep_to_write):
    TRoot=dataRoot

    sorted_one_component_DataFilesToRead=sort_data_files_by_flushEnd(summary_obs_name,component_name)

    one_component_StaringFileName=sorted_one_component_DataFilesToRead[startingFileInd]

    with open(one_component_StaringFileName,"rb") as fptr:
        one_component_inArrStart=np.array(pickle.load(fptr))

    one_component_Arr=one_component_inArrStart.reshape((sweep_to_write,-1))

    #read the rest of  pkl files
    for pkl_file in sorted_one_component_DataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            one_component_inArr=np.array(pickle.load(fptr))
            one_component_inArr=one_component_inArr.reshape((sweep_to_write,-1))
            one_component_Arr=np.concatenate((one_component_Arr,one_component_inArr),axis=0)


    one_component_ArrSelected=one_component_Arr[::lag,:]

    return one_component_ArrSelected

def save_oneComponent_dipole_data(one_component_ArrSelected,oneTStr,component_name,init_path):
    outCsvDataRoot=csv_out_path+"/csvOutAll/"
    outCsvFolder=outCsvDataRoot+f"/init_path{init_path}/"
    Path(outCsvFolder).mkdir(exist_ok=True,parents=True)
    outFileName=f"{component_name}.csv"
    outCsvFile=outCsvFolder+outFileName
    df=pd.DataFrame(one_component_ArrSelected)
    # Save to CSV
    print(f"saving {outCsvFile}")
    df.to_csv(outCsvFile, index=False, header=False)

t_save_start=datetime.now()
startingfileIndTmp,lagTmp,sweep_to_writeTmp=parseSummary(summary_obs_name)
if startingfileIndTmp<0:
    print("summary file does not exist for "+TStr+" "+summary_obs_name)
    exit(0)
component_Px="Px"
component_Py="Py"
Px_ArrSelected=one_dipole_component_extract_ForOneT(startingfileIndTmp,lagTmp,component_Px,sweep_to_writeTmp)
Py_ArrSelected=one_dipole_component_extract_ForOneT(startingfileIndTmp,lagTmp,component_Py,sweep_to_writeTmp)

save_oneComponent_dipole_data(Px_ArrSelected,TStr,component_Px,init_path)
save_oneComponent_dipole_data(Py_ArrSelected,TStr,component_Py,init_path)

t_save_End=datetime.now()
print(f"time: {t_save_End-t_save_start}")