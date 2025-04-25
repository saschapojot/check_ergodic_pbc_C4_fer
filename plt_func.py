import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
const_multiple=1000
alpha1=-1*const_multiple
alpha2=2*const_multiple
alpha3=-0.5*const_multiple


def H1(px,py):
    val1=alpha1*(px**2+py**2)
    val2=alpha2*(px**2+py**2)**2
    val3=alpha3*(px*py)**2
    return val1+val2+val3

# Generate a grid of points
x = np.linspace(-0.6, 0.6, 400)   # 400 points between -3 and 3
y = np.linspace(-0.6, 0.6, 400)
X, Y = np.meshgrid(x, y)

# Evaluate the function on the grid
Z = H1(X, Y)
magnitude = Z        # If you need the magnitude (absolute value)

# Plot using imshow
plt.figure(figsize=(8, 6))
plt.imshow(magnitude, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Magnitude of f(x, y)')
plt.title('Magnitude')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("func.png")

def H1_vec(p_vec):
    px,py=p_vec
    return H1(px,py)


# Provide an initial guess for the variables [x, y]
initial_guess = [-0.1, -0.2]
# Use the 'minimize' function from scipy.optimize
result = minimize(H1_vec, initial_guess, method='BFGS')

if result.success:
    print("Local minimum found at: ", result.x)
    print("Minimum value of H(x, y):", result.fun)
else:
    print("Optimization failed:", result.message)

# print(H1(1,1))