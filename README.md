#this project checks ergodic condition of a C4 ferroelectric system under PBC
# the lattice is square!

python mk_dir.py, to set coefficients, T, and directories

##########################################
To manually perform each step of computations for U
1. python launch_one_run_U.py ./path/to/mc.conf
2. make run_mc
3. ./run_mc ./path/to/cppIn.txt
4. python check_after_one_run_U.py ./path/to/mc.conf  startingFileIndSuggest
5. go to 1, until no more data points are needed