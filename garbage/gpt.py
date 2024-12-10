import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# Define the piecewise linear, continuous, but non-differentiable function
def piecewise_function(x1, x2, m=14):
    result = np.zeros_like(x1)
    # Define the size of each interval
    interval_size = 1 / m

    # Loop over each interval
    for i in range(m):
        for j in range(m):
            # Define the bounds of the current interval
            x1_min, x1_max = i * interval_size, (i + 1) * interval_size
            x2_min, x2_max = j * interval_size, (j + 1) * interval_size
            
            # Piecewise linear function within this interval
            try:
                mask = (x1 >= x1_min) & (x1 <= x1_max) & (x2 >= x2_min) & (x2 <= x2_max)
                result[mask] = np.where(
                    (x1[mask] + x2[mask]) <= (x1_min + x2_min + interval_size),
                    2 * (x1[mask] - x1_min) + 2 * (x2[mask] - x2_min),
                    2 * (x1_max - x1[mask]) + 2 * (x2_max - x2[mask])
                )
            except:
                continue
    
    return result

def piecewise_tensor(X):
    output = piecewise_function(X[:,0].numpy(), X[:,1].numpy(), m=8)
    return torch.tensor(output, dtype = torch.float32)
    #max_j = 10
# Compute the function values

x = np.linspace(0, 1, 100)  # Example range for x
y = np.linspace(0, 1, 100)  # Example range for y

    # Generate meshgrid
X, Y = np.meshgrid(x, y)

    # Flatten the two meshgrid arrays
X_flat = X.flatten()
Y_flat = Y.flatten()

# Concatenate into a single array of shape (x, 2), where x is the number of points
mesh_data = np.stack((X_flat, Y_flat), axis=1)
x = torch.tensor(mesh_data)
output = piecewise_tensor(x)
Z = output.reshape(100, 100)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Plot of Piecewise Function')

# Show the plot
plt.show()