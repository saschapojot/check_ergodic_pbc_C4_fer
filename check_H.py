
import pickle
from math import factorial
import numpy as np

Px_file="./dataAll/N5/T5.81/init_path0/U_dipole_dataFiles/Px/Px_init.pkl"


Py_file="./dataAll/N5/T5.81/init_path0/U_dipole_dataFiles/Py/Py_init.pkl"
N0=5
N1=5

with open(Px_file,"rb") as fptr:
    px_arr=np.array(pickle.load(fptr))

with open(Py_file,"rb") as fptr:
    py_arr=np.array(pickle.load(fptr))

ind1=18
ind2=14
const_multiple=1000
alpha1_coef=1/(factorial(1)*factorial(4))
alpha1=alpha1_coef*const_multiple
#alpha2: 1/(1!2!)
alpha2_coef=1/(factorial(1)*factorial(2))
alpha2=alpha2_coef*const_multiple
#alpha3: 1/(2!4!)
alpha3_coef=1/(factorial(2))
alpha3=alpha3_coef*const_multiple
J=1/5*alpha1

pn0n1=np.array([px_arr[ind1],py_arr[ind1]])

pm0m1=np.array([px_arr[ind2],py_arr[ind2]])

a=2
def double_ind_to_flat_ind( n0, n1):
    return n0 * N1 + n1

def H1(n0,n1):
    flat_ind=double_ind_to_flat_ind(n0,n1)
    px_n0n1=px_arr[flat_ind]
    py_n0n1=py_arr[flat_ind]
    val1=alpha1*((px_n0n1**2-py_n0n1**2)**2-4*px_n0n1**2*py_n0n1**2)
    val2=alpha2*px_n0n1*py_n0n1*(px_n0n1**2-py_n0n1**2)

    val3=alpha3*(px_n0n1**2+py_n0n1**2)
    return val1+val2+val3


def H2(n0,n1,ind_diff0,ind_diff1):
    r_diff=np.array([ind_diff0,ind_diff1])*a
    m0=(n0+ind_diff0)%N0

    m1=(n1+ind_diff1)%N1

    flat_ind_left=double_ind_to_flat_ind(n0,n1)

    flat_ind_right=double_ind_to_flat_ind(m0,m1)

    pxn0n1=px_arr[flat_ind_left]
    pyn0n1=py_arr[flat_ind_left]
    pn0n1=np.array([pxn0n1,pyn0n1])

    pxm0m1=px_arr[flat_ind_right]
    pym0m1=py_arr[flat_ind_right]
    pm0m1=np.array([pxm0m1,pym0m1])

    val1=J/np.linalg.norm(r_diff,2)**2*np.dot(pn0n1,pm0m1)

    val2=-2*J/np.linalg.norm(r_diff,2)**4*np.dot(pn0n1,r_diff)*np.dot(pm0m1,r_diff)

    return val1+val2


part1=0
part2=0
for n0 in range(0,N0):
    for n1 in range(0,N1):
        part1+=H1(n0,n1)

for n0 in range(0,N0):
    for n1 in range(0,N1):
        for ind_diff0 in [-1,0,1]:
            for ind_diff1 in [-1,0,1]:
                if ind_diff0==0 and ind_diff1==0:
                    continue
                else:
                    part2+=H2(n0,n1,ind_diff0,ind_diff1)

print(f"part1={part1}")
print(f"part2={part2}")
e=part1+1/2*part2
print(e)