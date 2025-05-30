import re
import subprocess
import sys

import json
argErrCode=2

if (len(sys.argv)!=2):
    print("wrong number of arguments")
    print("example: python launch_one_run.py /path/to/mc.conf")
    exit(argErrCode)

confFileName=str(sys.argv[1])
invalidValueErrCode=1
summaryErrCode=5
loadErrCode=3
confErrCode=4


#################################################
#parse conf, get jsonDataFromConf
confResult=subprocess.run(["python3", "./init_run_scripts/parseConf.py", confFileName], capture_output=True, text=True)
confJsonStr2stdout=confResult.stdout
# print(confJsonStr2stdout)

if confResult.returncode !=0:
    print("Error running parseConf.py with code "+str(confResult.returncode))
    # print(confResult.stderr)
    exit(confErrCode)
match_confJson=re.match(r"jsonDataFromConf=(.+)$",confJsonStr2stdout)
if match_confJson:
    jsonDataFromConf=json.loads(match_confJson.group(1))
else:
    print("jsonDataFromConf missing.")
    exit(confErrCode)
# print(jsonDataFromConf)

##################################################

#read summary file, get jsonFromSummary
# read dipole summary
parseSummaryResult=subprocess.run(["python3","./init_run_scripts/search_and_read_summary_dipole.py", json.dumps(jsonDataFromConf)],capture_output=True, text=True)
# print(parseSummaryResult.stdout)
if parseSummaryResult.returncode!=0:
    print("Error in parsing summary with code "+str(parseSummaryResult.returncode))
    # print(parseSummaryResult.stdout)
    # print(parseSummaryResult.stderr)
    exit(summaryErrCode)

match_summaryJson=re.match(r"jsonFromSummary=(.+)$",parseSummaryResult.stdout)
if match_summaryJson:
    jsonFromSummary=json.loads(match_summaryJson.group(1))
# print(jsonFromSummary)

##################################################


###############################################
#load previous data, to get paths
#get loadedJsonData

loadResult=subprocess.run(["python3","./init_run_scripts/load_previous_data.py", json.dumps(jsonDataFromConf), json.dumps(jsonFromSummary)],capture_output=True, text=True)

# print(loadResult.stdout)
if loadResult.returncode!=0:
    print("Error in loading with code "+str(loadResult.returncode))
    exit(loadErrCode)

match_loadJson=re.match(r"loadedJsonData=(.+)$",loadResult.stdout)
if match_loadJson:
    loadedJsonData=json.loads(match_loadJson.group(1))
else:
    print("loadedJsonData missing.")
    exit(loadErrCode)

# print(f"loadedJsonData={loadedJsonData}")
###############################################

###############################################
#construct parameters that are passed to mc
TStr=jsonDataFromConf["T"]
aStr=jsonDataFromConf["a"]
JStr=jsonDataFromConf["J"]
NStr=jsonDataFromConf["N"]
qStr=jsonDataFromConf["q"]
N_half_sideStr=jsonDataFromConf["N_half_side"]
alpha1Str=jsonDataFromConf["alpha1"]
alpha2Str=jsonDataFromConf["alpha2"]
alpha3Str=jsonDataFromConf["alpha3"]

sweep_to_write=jsonDataFromConf["sweep_to_write"]
flushLastFile=loadedJsonData["flushLastFile"]
newFlushNum=jsonFromSummary["newFlushNum"]
TDirRoot=jsonFromSummary["TDirRoot"]
U_dipole_dataDir=jsonFromSummary["U_dipole_dataDir"]

hStr=jsonDataFromConf["h"]
sweep_multipleStr=jsonDataFromConf["sweep_multiple"]

init_path=jsonDataFromConf["init_path"]

params2cppInFile=[
    TStr+"\n",
    aStr+"\n",
    JStr+"\n",
    NStr+"\n",
    qStr+"\n",
    alpha1Str+"\n",
    alpha2Str+"\n",
    alpha3Str+"\n",
    sweep_to_write+"\n",
    newFlushNum+"\n",
    flushLastFile+"\n",
    TDirRoot+"\n",
    U_dipole_dataDir+"\n",
    hStr+"\n",
    sweep_multipleStr+"\n",
    N_half_sideStr+"\n",
    init_path+"\n"


    ]



cppInParamsFileName=TDirRoot+"/cppIn.txt"
with open(cppInParamsFileName,"w+") as fptr:
    fptr.writelines(params2cppInFile)