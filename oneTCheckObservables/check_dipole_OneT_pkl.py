import numpy as np
from datetime import datetime
import statsmodels.api as sm
import sys
import re
import warnings


from scipy.stats import ks_2samp
import glob

import os
import json
import pickle

#This script checks if total dipole values reach equilibrium and writes summary file of dipole
#This file checks pkl files
argErrCode=2
sameErrCode=3
missingErrCode=4
eps_for_auto_corr=5e-2

jsonFromSummaryLast=json.loads(sys.argv[1])
jsonDataFromConf=json.loads(sys.argv[2])

startingFileIndSuggest=int(sys.argv[3])
TDirRoot=jsonFromSummaryLast["TDirRoot"]
U_dipole_dataDir=jsonFromSummaryLast["U_dipole_dataDir"]
effective_data_num_required=int(jsonDataFromConf["effective_data_num_required"])

N=int(jsonDataFromConf["N"])
sweep_to_write=int(jsonDataFromConf["sweep_to_write"])
summary_dipole_File=TDirRoot+"/summary_dipole.txt"

def sort_data_files_by_flushEnd(oneDir):
    dataFilesAll=[]
    flushEndAll=[]
    for oneDataFile in glob.glob(oneDir+"/flushEnd*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"flushEnd(\d+)",oneDataFile)
        if matchEnd:
            indTmp=int(matchEnd.group(1))
            flushEndAll.append(indTmp)

    endInds=np.argsort(flushEndAll)
    sortedDataFiles=[dataFilesAll[i] for i in endInds]
    return sortedDataFiles


def parseSummary_dipole():
    startingFileInd=-1
    summaryFileExists=os.path.isfile(summary_dipole_File)
    if summaryFileExists==False:
        return startingFileInd
    with open(summary_dipole_File,"r") as fptr:
        lines=fptr.readlines()
    for oneLine in lines:
        #match startingFileInd
        matchStartingFileInd=re.search(r"startingFileInd=(\d+)",oneLine)
        if matchStartingFileInd:
            startingFileInd=int(matchStartingFileInd.group(1))
    return startingFileInd


def auto_corrForOneVec(vec):
    """

    :param colVec: a vector of data
    :return:
    """
    same=False
    NLags=int(len(vec)*3/4)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
    try:
        acfOfVec=sm.tsa.acf(vec,nlags=NLags)
    except Warning as w:
        same=True

    acfOfVecAbs=np.abs(acfOfVec)
    minAutc=np.min(acfOfVecAbs)
    lagVal=-1
    if minAutc<=eps_for_auto_corr:
        lagVal=np.where(acfOfVecAbs<=eps_for_auto_corr)[0][0]

    return same,lagVal


def ksTestOneColumn(vec,lag):
    """

    :param vec: a vector of data
    :param lag: auto-correlation length
    :return:
    """
    vecSelected=vec[::lag]
    lengthTmp=len(vecSelected)
    if lengthTmp%2==1:
        lengthTmp-=1

    lenPart=int(lengthTmp/2)
    vecToCompute=vecSelected[-lengthTmp:]

    #ks test
    selectedVecPart0=vecToCompute[:lenPart]
    selectedVecPart1=vecToCompute[lenPart:]
    result=ks_2samp(selectedVecPart0,selectedVecPart1)

    return result.pvalue,result.statistic, lenPart*2

def check_DipoleDataFilesForOneT(Px_dir,Py_dir):
    Px_sortedDataFilesToRead=sort_data_files_by_flushEnd(Px_dir)
    # print(f"Px_sortedDataFilesToRead={Px_sortedDataFilesToRead}")
    Py_sortedDataFilesToRead=sort_data_files_by_flushEnd(Py_dir)

    len_Px=len(Px_sortedDataFilesToRead)

    len_Py=len(Py_sortedDataFilesToRead)
    diff2=(len_Px-len_Py)**2

    if diff2>0:
        print(f"diff2={diff2}, data missing.")
        exit(missingErrCode)
    startingFileInd=parseSummary_dipole()
    if startingFileInd<0:
        #we guess that the equilibrium starts at this file
        startingFileInd=startingFileIndSuggest

    Px_startingFileName=Px_sortedDataFilesToRead[startingFileInd]

    Py_startingFileName=Py_sortedDataFilesToRead[startingFileInd]


    #read Px
    with open(Px_startingFileName,"rb") as fptr:
        Px_inArrStart=np.array(pickle.load(fptr))
    Px_arr=Px_inArrStart.reshape((sweep_to_write,-1))

    #read the rest of Px pkl files
    for pkl_file in Px_sortedDataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            Px_inArr=np.array(pickle.load(fptr))
        Px_inArr=Px_inArr.reshape((sweep_to_write,-1))
        Px_arr=np.concatenate((Px_arr,Px_inArr),axis=0)

    #read Py
    with open(Py_startingFileName,"rb") as fptr:
        Py_inArrStart=np.array(pickle.load(fptr))

    Py_arr=Py_inArrStart.reshape((sweep_to_write,-1))
    #read the rest of Py pkl files
    for pkl_file in Py_sortedDataFilesToRead[(startingFileInd+1):]:
        with open(pkl_file,"rb") as fptr:
            Py_inArr=np.array(pickle.load(fptr))
        Py_inArr=Py_inArr.reshape((sweep_to_write,-1))
        Py_arr=np.concatenate((Py_arr,Py_inArr),axis=0)

    dipole_x=Px_arr
    dipole_y=Py_arr
    polarization_x=np.mean(dipole_x,axis=1)
    polarization_y=np.mean(dipole_y,axis=1)
    sameTmp_polarization_x,lagTmp_polarization_x=auto_corrForOneVec(polarization_x)
    sameTmp_polarization_y,lagTmp_polarization_y=auto_corrForOneVec(polarization_y)

    same_vec=[sameTmp_polarization_x,sameTmp_polarization_y]
    lag_vec=[lagTmp_polarization_x,lagTmp_polarization_y]
    if any(same_vec) or -1 in lag_vec:
        return [-2],[-1],-1,[],-1
    lagMax=np.max(lag_vec)
    pTmp_polarization_x,statTmp_polarization_x,lengthTmp_polarization_x=ksTestOneColumn(polarization_x,lagMax)

    pTmp_polarization_y,statTmp_polarization_y,lengthTmp_polarization_y=ksTestOneColumn(polarization_y,lagMax)

    pVec=[pTmp_polarization_x,pTmp_polarization_y]

    statVec=[statTmp_polarization_x,statTmp_polarization_y]
    numDataPoints=lengthTmp_polarization_x
    return pVec,statVec,numDataPoints,lag_vec,startingFileInd


min_data_num_for_stats=40

Px_dir=U_dipole_dataDir+"/Px/"
Py_dir=U_dipole_dataDir+"/Py/"

pVec,statVec,numDataPoints,lag_vec,startingFileInd=check_DipoleDataFilesForOneT(Px_dir,Py_dir)
lagMax=np.max(lag_vec)
print(f"TDirRoot={TDirRoot}")
print("lagMax="+str(lagMax))
print("numDataPoints="+str(numDataPoints))
print(f"pVec={pVec}")
print(f"statVec={statVec}")
statThreshhold=0.1
if pVec[0]==-2:
    with open(summary_dipole_File,"w+") as fptr:
        msg="error: same\n"
        fptr.writelines(msg)
        exit(sameErrCode)


if numDataPoints<0:
    msg="high correlation"
    with open(summary_dipole_File,"w+") as fptr:
        fptr.writelines(msg)
    exit(0)


if (np.min(pVec)>=0.01 or np.max(statVec)<=statThreshhold) and numDataPoints>=min_data_num_for_stats:
    if numDataPoints>=effective_data_num_required:
        newDataPointNum=0
    else:
        newDataPointNum=effective_data_num_required-numDataPoints

    msg="equilibrium\n" \
        +"lag="+str(lagMax)+"\n" \
        +"numDataPoints="+str(numDataPoints)+"\n" \
        +"startingFileInd="+str(startingFileInd)+"\n" \
        +"newDataPointNum="+str(newDataPointNum)+"\n" \
        +"sweep_to_write="+str(sweep_to_write)+"\n"

    print(msg)
    with open(summary_dipole_File,"w+") as fptr:
        fptr.writelines(msg)
    exit(0)
#continue
continueMsg="continue\n"
if not (np.min(pVec)>=0.01 or np.max(statVec)<=statThreshhold):
    continueMsg+="stat value: "+str(np.max(statVec))+"\n"
    continueMsg+="p value: "+str(np.min(pVec))+"\n"


if numDataPoints<min_data_num_for_stats:
    #not enough data number

    continueMsg+="numDataPoints="+str(numDataPoints)+" too low\n"
    continueMsg+="lag="+str(lagMax)+"\n"
print(continueMsg)
with open(summary_dipole_File,"w+") as fptr:
    fptr.writelines(continueMsg)
exit(0)
