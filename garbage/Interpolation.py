import torch
import numpy as np 
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Step 1: Define the original function f(x, y) (continuous)
def f(x, y):
    """Example continuous function."""
    #return np.exp(np.sin(x**2 + y**2) + np.exp(y**2))
    epsilon = 0.001
    return np.sin(1 / (x**2 + y**2 + epsilon)) * np.cos(20 * np.pi * x) * np.cos(20 * np.pi * y)
# Step 2: Create a grid of points for interpolation
def create_grid(n=10):
    """Creates a grid of points over the domain [0,1] x [0,1] with n points in each direction."""
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)
    return X.ravel(), Y.ravel()

print(f(-0.0075,  0.5547))
# Step 3: Define the function to perform interpolation
def interpolate_function(f, x_points, y_points):
    """Interpolates the function f over a given set of x and y points using piecewise-linear interpolation."""
    # Create the values of f at the grid points
    z_points = f(x_points, y_points)
    points = np.column_stack((x_points, y_points))
    
    # Perform linear interpolation over the grid points
    interpolator = LinearNDInterpolator(points, z_points)
    
    # Return a callable interpolated function
    def f_inter(x, y):
        return interpolator(x, y)
    
    return f_inter

# Step 4: Evaluate and calculate sup norm of the difference
def calculate_sup_norm(f, f_inter, test_points):
    """Evaluates the original and interpolated functions and computes the sup norm (maximum error)."""
    # Evaluate the original function
    f_values = f(test_points[:, 0], test_points[:, 1])
    
    # Evaluate the interpolated function
    f_inter_values = f_inter(test_points[:, 0], test_points[:, 1])
    
    # Compute the absolute difference
    difference = np.abs(f_values - f_inter_values)
    
    # Supremum norm (max absolute difference)
    sup_norm = np.max(difference)
    
    # Find the point of greatest divergence
    max_diff_index = np.argmax(difference)
    point_of_max_divergence = test_points[max_diff_index]
    
    return sup_norm, point_of_max_divergence
def func_inter(X):
    func = f
    x_grid, y_grid = create_grid(n=100)
    # 2. Define the interpolated function f_inter
    f_inter = interpolate_function(func, x_grid, y_grid)
    out = f_inter(X[:,0].detach().numpy(), X[:,1].detach().numpy())
    return torch.tensor(out, dtype = torch.float32)

# 2. Define the interpolated function f_inter
# Example Usage
# 1. Create a grid of points for interpolation
x_grid, y_grid = create_grid(n=100)

# 2. Define the interpolated function f_inter
f_inter = interpolate_function(f, x_grid, y_grid)

# 3. Generate test points for evaluating the sup norm
test_x, test_y = create_grid(n=1000)  # Finer grid for testing
test_points = np.column_stack((test_x, test_y))

# 4. Evaluate the sup norm and point of greatest divergence
sup_norm, point_of_max_divergence = calculate_sup_norm(f, f_inter, test_points)

# Output results
print(f"Supremum norm (max divergence): {sup_norm}")
print(f"Point of greatest divergence: {point_of_max_divergence}")

# Example: Call the interpolated function for any point (x, y)
x_test, y_test = 0.5, 0.5
print(f"Interpolated value at ({x_test}, {y_test}): {f_inter(x_test, y_test)}")
num_points = 50
x = np.linspace(-1, 1, num_points)  # Example range for x
y = np.linspace(-1, 1, num_points)  # Example range for y

    # Generate meshgrid
X, Y = np.meshgrid(x, y)

    # Flatten the two meshgrid arrays
X_flat = X.flatten()
Y_flat = Y.flatten()

    # Concatenate into a single array of shape (x, 2), where x is the number of points
mesh_data = np.stack((X_flat, Y_flat), axis=1)
x = mesh_data
output = f_inter(X,Y)
out_2 = f(X, Y)
Z = output.reshape(num_points, num_points)
Z2 = out_2.reshape(num_points,num_points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('f Interpolated')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z2, cmap='viridis')

# Add labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('f')
# Show the plot
plt.show()
x = np.linspace(0, 1, 50)  # Example range for x
y = np.linspace(0, 1, 50)  # Example range for y

    # Generate meshgrid
X, Y = np.meshgrid(x, y)

    # Flatten the two meshgrid arrays
X_flat = X.flatten()
Y_flat = Y.flatten()

    # Concatenate into a single array of shape (x, 2), where x is the number of points
mesh_data = np.stack((X_flat, Y_flat), axis=1)
x = torch.tensor(mesh_data)
output = func_inter(x)
Z = output.reshape(50, 50)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Plot of Piecewise Function from tensor')

# Show the plot
plt.show()
