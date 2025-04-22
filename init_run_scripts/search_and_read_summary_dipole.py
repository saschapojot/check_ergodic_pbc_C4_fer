import re
from decimal import Decimal
import json
import sys
from pathlib import Path
import os
import shutil
import numpy as np

#this script initializes the parameters for mc computation by reading dipole summary file
#and also creates/erases necessary folders

invalidValueErrCode=1
mcErrCode=2
pathErrCode=3
numArgErr=4

if (len(sys.argv)!=2):
    print("wrong number of arguments.")
    exit(numArgErr)

jsonDataFromConf=json.loads(sys.argv[1])
#read json
T=float(jsonDataFromConf["T"])
if T<=0:
    print("invalid temperature: "+str(T))
    exit(invalidValueErrCode)


def str_2_bool(string):
    if string=="True":
        return True
    elif string=="False":
        return False
    else:
        print("format error: "+str(string))
        exit(invalidValueErrCode)


erase_data_if_exist=str_2_bool(jsonDataFromConf["erase_data_if_exist"])

search_and_read_summary_file=str_2_bool(jsonDataFromConf["search_and_read_summary_file"])
aStr=jsonDataFromConf["a"]
a=float(aStr)
if a<=0:
    print("a:"+str(a)," a should be >0")
    exit(invalidValueErrCode)


effective_data_num_required=int(jsonDataFromConf["effective_data_num_required"])

sweep_to_write=int(jsonDataFromConf["sweep_to_write"])

default_flush_num=int(jsonDataFromConf["default_flush_num"])

sweep_multiple=int(jsonDataFromConf["sweep_multiple"])

confFileName=jsonDataFromConf["confFileName"]


#TPDirRoot contains everything for a computation
TDirRoot=os.path.dirname(confFileName)

TDirRoot=TDirRoot+"/"

#create directory for raw data of U and dipole
U_dipole_dataDir=TDirRoot+"/U_dipole_dataFiles/"

# print(dataDir)
if erase_data_if_exist==True:
    if os.path.isdir(U_dipole_dataDir):
        try:
            shutil.rmtree(U_dipole_dataDir)
            print(f'Directory {U_dipole_dataDir} and all its contents have been removed successfully')
        except OSError as e:
            print(f'Error: {U_dipole_dataDir} : {e.strerror}')
            exit(pathErrCode)


#create dataDir if not exists
Path(U_dipole_dataDir).mkdir(exist_ok=True, parents=True)
swpNumInOneFlush=sweep_to_write*sweep_multiple

#parameters to guide mc computation
lag=-1
startingFileInd=-1
newDataPointNum=-1
newMcStepNum=swpNumInOneFlush*default_flush_num
newFlushNum=default_flush_num

def create_jsonFromSummary(startingFileIndVal,newMcStepNumVal,
                           newDataPointNumVal,newFlushNumVal,TDirRootStr,U_dipole_dataDirStr):

    """

    :param startingFileIndVal:
    :param newMcStepNumVal:
    :param newDataPointNumVal:
    :param newFlushNumVal:
    :param TDirRootStr:
    :param U_dipole_dataDirStr:
    :return: jsonFromSummary as string
    """
    outDict={
        "startingFileInd":str(startingFileIndVal),
        "newMcStepNum":str(newMcStepNumVal),
        "newDataPointNum":str(newDataPointNumVal),
        "newFlushNum": str(newFlushNumVal),
        "TDirRoot":str(TDirRootStr),
        "U_dipole_dataDir": str(U_dipole_dataDirStr),
    }

    return json.dumps(outDict)


summaryFileName_dipole=TDirRoot+"/summary_dipole.txt"
summaryFileExists= os.path.isfile(summaryFileName_dipole)

#if summary file does not exist, return -1, sweep_to_write*default_flush_num, then exit with code 0
if summaryFileExists==False:
    jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,newMcStepNum,
                                              newDataPointNum,newFlushNum,TDirRoot,U_dipole_dataDir)
    jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
    print(jsonFromSummary_stdout)
    exit(0)

#parse summary file
with open(summaryFileName_dipole,"r") as fptr:
    linesInSummaryFile= fptr.readlines()

for oneLine in linesInSummaryFile:
    matchErr=re.search(r"error",oneLine)
    #if "error" is matched
    if matchErr:
        print("error in previous computation, please re-run.")
        exit(mcErrCode)
    #if "continue" is matched
    matchContinue=re.search(r"continue",oneLine)
    if matchContinue:
        jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,newMcStepNum,
                                                  newDataPointNum,newFlushNum,TDirRoot,U_dipole_dataDir)
        jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
        print(jsonFromSummary_stdout)
        exit(0)

    #if "high" is matched
    matchHigh=re.search(r"high",oneLine)
    if matchHigh:
        jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,newMcStepNum,
                                                  newDataPointNum,newFlushNum,TDirRoot,U_dipole_dataDir)
        jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
        print(jsonFromSummary_stdout)
        exit(0)

    #the rest of the cases is "equilibrium"
    matchEq=re.search(r"equilibrium",oneLine)
    if matchEq:
        continue

    #match lag
    matchLag=re.match(r"lag\s*=\s*(\d+)",oneLine)
    if matchLag:
        lag=int(matchLag.group(1))


    #match newDataPointNum
    matchNew=re.match(r"newDataPointNum\s*=\s*(\d+)",oneLine)
    if matchNew:
        newDataPointNum=int(matchNew.group(1))

    #match startingFileInd
    matchStartingFileInd=re.match(r"startingFileInd\s*=\s*(\d+)",oneLine)
    if matchStartingFileInd:
        startingFileInd=int(matchStartingFileInd.group(1))


newMcStepNum=lag*newDataPointNum
newFlushNum=int(np.ceil(newMcStepNum/(sweep_to_write)))
jsonFromSummaryStr=create_jsonFromSummary(startingFileInd,newMcStepNum,
                                          newDataPointNum,newFlushNum,TDirRoot,U_dipole_dataDir)
jsonFromSummary_stdout="jsonFromSummary="+jsonFromSummaryStr
print(jsonFromSummary_stdout)
exit(0)