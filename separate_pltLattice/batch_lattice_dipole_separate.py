import subprocess
import signal
import sys
import os
import re
import glob
from datetime import datetime
import numpy as np

#this script executes oneT_one_path_lattice_dipole.py
# Ensure usage: the script expects one command line argument
if len(sys.argv) != 3:
    print("wrong number of arguments")
    sys.exit(1)

N = int(sys.argv[1])
TStr=sys.argv[2]

dataRoot = f"../dataAll/N{N}/T{TStr}/csvOutAll/"
path_dir_vec=[]
path_name_vec=[]
pattern_path = re.compile(r'(?:init_path(\d+|main))')

for path_dir in glob.glob(dataRoot+"/init*"):
    path_dir_vec.append(path_dir)
    match_path_name=pattern_path.search(path_dir)
    path_name_vec.append(match_path_name.group(1))


tStart = datetime.now()
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
            ["python3", "-u", "oneT_one_path_lattice_dipole.py", f"{N}",TStr,one_path_name],
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