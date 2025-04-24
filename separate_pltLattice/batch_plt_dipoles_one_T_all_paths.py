import subprocess
import signal
import sys
import os
import re
import glob
from datetime import datetime
import numpy as np

#this script executes load_csv_one_T_cone_path_dipole
#for all paths, 1T

if len(sys.argv) != 3:
    print("wrong number of arguments")
    sys.exit(1)

N = int(sys.argv[1])
TStr = sys.argv[2]

dataRoot = f"../dataAll/N{N}/T{TStr}/csvOutAll/"

path_dir_vec=[]
path_name_vec=[]
pattern_path = re.compile(r'(?:init_path(\d+|main))')

for path_dir in glob.glob(dataRoot+"/init*"):
    path_dir_vec.append(path_dir)
    match_path_name=pattern_path.search(path_dir)
    path_name_vec.append(match_path_name.group(1))

dipole_each_site_dir=dataRoot+"/dipole_each_site/"
# Create the directory if it doesn't exist
if not os.path.exists(dipole_each_site_dir):
    try:
        os.makedirs(dipole_each_site_dir)
        print(f"Directory '{dipole_each_site_dir}' created successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"Directory '{dipole_each_site_dir}' already exists.")


tStart = datetime.now()
# Function to terminate the subprocess
def terminate_process(proc):
    try:
        # Terminate the whole process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        print(f"Terminated subprocess with PID {proc.pid}")
    except Exception as e:
        print(f"Error terminating subprocess with PID {proc.pid}: {e}")


try:
    for k in range(0,len(path_name_vec)):
        one_path_name=path_name_vec[k]
        print(f"Running subprocess for {one_path_name}")
        stats_process = subprocess.Popen(
            ["python3", "-u", "load_csv_one_T_one_path_dipole.py", f"{N}",TStr,one_path_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True  # This makes it easier to kill the process group
        )
        try:
            # Read standard output line by line
            while True:
                output = stats_process.stdout.readline()
                if output == '' and stats_process.poll() is not None:
                    break
                if output:
                    print(output.strip())

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Terminating current subprocess...")
            terminate_process(stats_process)
            sys.exit(1)
        # Final communication in case there's remaining output
        stdout, stderr = stats_process.communicate()
        if stdout:
            print(stdout.strip())
        if stderr:
            print(stderr.strip())

except KeyboardInterrupt:
    # If a KeyboardInterrupt occurs outside of the inner loop, ensure all running subprocesses are terminated.
    print("KeyboardInterrupt detected in the main loop. Exiting.")
    sys.exit(1)

tEnd = datetime.now()
print(f"total time:{ tEnd-tStart}")