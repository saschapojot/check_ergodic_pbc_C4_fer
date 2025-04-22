import sys
import glob
import re
import json
from decimal import Decimal, getcontext
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import pickle


#this script loads previous data
# np.random.seed(29)
numArgErr=4
valErr=5


if (len(sys.argv)!=3):
    print("wrong number of arguments.")
    exit(numArgErr)

jsonDataFromConf =json.loads(sys.argv[1])
jsonFromSummary=json.loads(sys.argv[2])

U_dipole_dataDir=jsonFromSummary["U_dipole_dataDir"]

startingFileInd=jsonFromSummary["startingFileInd"]



NStr=jsonDataFromConf["N"]

N=int(NStr)
if N<=0:
    print("invalid N="+str(N))
    exit(valErr)


a=float(jsonDataFromConf["a"])
if a<=0:
    print(f"invalid a={a}")
    exit(valErr)


q=float(jsonDataFromConf["q"])
if q<=0:
    print(f"invalid q={a}")
    exit(valErr)


#search and read U_dipole files

d_max=1/7*a
d_min=-1/7*a
#search flushEnd
pklFileList=[]
flushEndAll=[]


#read Px files
for file in glob.glob(U_dipole_dataDir+"/Px/flushEnd*.pkl"):
    pklFileList.append(file)
    matchEnd=re.search(r"flushEnd(\d+)",file)
    if matchEnd:
        flushEndAll.append(int(matchEnd.group(1)))


flushLastFile=-1
def format_using_decimal(value, precision=15):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)


def create_init_Px_Py(U_dipole_dataDir):

    #Px
    outPath_Px=U_dipole_dataDir+"/Px/"
    Path(outPath_Px).mkdir(exist_ok=True,parents=True)
    outFileName_Px=outPath_Px+"/Px_init.pkl"
    Px_init_mat=q*np.array([np.random.uniform(d_min,d_max) for i in range(0,N*N)])
    with open(outFileName_Px,"wb") as fptr:
        pickle.dump(Px_init_mat,fptr)

    #Py
    outPath_Py=U_dipole_dataDir+"/Py/"
    Path(outPath_Py).mkdir(exist_ok=True,parents=True)
    outFileName_Py=outPath_Py+"/Py_init.pkl"
    Py_init_mat=q*np.array([np.random.uniform(d_min,d_max) for i in range(0,N*N)])
    with open(outFileName_Py,"wb") as fptr:
        pickle.dump(Py_init_mat,fptr)



def create_loadedJsonData(flushLastFileVal):

    initDataDict={

        "flushLastFile":str(flushLastFileVal)
    }
    # print(initDataDict)
    return json.dumps(initDataDict)


#if no data found, return flush=-1
if len(pklFileList)==0:
    create_init_Px_Py(U_dipole_dataDir)
    out_U_path=U_dipole_dataDir+"/U/"
    Path(out_U_path).mkdir(exist_ok=True,parents=True)
    loadedJsonDataStr=create_loadedJsonData(flushLastFile)
    loadedJsonData_stdout="loadedJsonData="+loadedJsonDataStr
    print(loadedJsonData_stdout)
    exit(0)


#if found pkl data with flushEndxxxx
sortedEndInds=np.argsort(flushEndAll)
sortedflushEnd=[flushEndAll[ind] for ind in sortedEndInds]
loadedJsonDataStr=create_loadedJsonData(sortedflushEnd[-1])
loadedJsonData_stdout="loadedJsonData="+loadedJsonDataStr
print(loadedJsonData_stdout)
exit(0)