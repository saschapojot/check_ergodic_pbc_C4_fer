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

##########################################
To manually perform each step of computations for dipole
1. python launch_one_run_dipole.py ./path/to/mc.conf
2. make run_mc
3. ./run_mc ./path/to/cppIn.txt
4. python check_after_one_run_dipole.py ./path/to/mc.conf  startingFileIndSuggest
5. go to 1, until no more data points are needed

#########################################
To run 1 pass of mc with checking statistics of dipole
1. cmake .
2. make run_mc
3. python exec_checking_dipole.py T N startingFileIndSuggest
4. run 3 until equilibrium
5. python exec_noChecking.py T N