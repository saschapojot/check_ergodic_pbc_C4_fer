import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
import scipy.stats as stats
import os
from decimal import Decimal, getcontext
errFileNotExist=1

#this script converts Px, Py csv files to average
#for one T, one computation path

if (len(sys.argv)!=4):
    print("wrong number of arguments")
    exit()

N=int(sys.argv[1])
TStr=sys.argv[2]
init_path=sys.argv[3]


csvDataFolderRoot=f"../dataAll/N{N}/T{TStr}/csvOutAll/init_path{init_path}/"

Px_csv_file=csvDataFolderRoot+"/Px.csv"


Py_csv_file=csvDataFolderRoot+"/Py.csv"

if not os.path.exists(Px_csv_file):
    print(f"Px does not exist for {TStr}")
    exit(errFileNotExist)


if not os.path.exists(Py_csv_file):
    print(f"Py does not exist for {TStr}")
    exit(errFileNotExist)


Px_csv_arr=np.array(pd.read_csv(Px_csv_file,header=None))

Px_avg=np.mean(Px_csv_arr,axis=0)


Py_csv_arr=np.array(pd.read_csv(Py_csv_file,header=None))

Py_avg=np.mean(Py_csv_arr,axis=0)

out_dipole_file_name=csvDataFolderRoot+"/avg_dipole_combined.csv"

out_arr=np.array([
    Px_avg,Py_avg
])
df=pd.DataFrame(out_arr)
df.to_csv(out_dipole_file_name, header=False, index=False)