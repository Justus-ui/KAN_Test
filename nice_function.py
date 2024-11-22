import numpy as np
import matplotlib.pyplot as plt
import torch
# Definition of the sum function f(j)
def f_j(j):
    return 1 - (1/2)**j

# Piecewise function definition
def triangle_function(x, max_j=10):
    for j in range(1, max_j + 1):
        f_j_start = f_j(j -1)
        f_j_end = f_j(j)
        midpoint = (f_j_start + f_j_end) / 2
        
        if f_j_start <= x <= f_j_end:
            if x <= midpoint:
                return 2 * (x - f_j_start) / (f_j_end - f_j_start) *j#* (1/j )
            else:
                return 2 * (f_j_end - x) / (f_j_end - f_j_start) *j#* (1/j )
    return 0

def triangle_tensor(X):
    #max_j = 10
    vectorized_function = np.vectorize(triangle_function)
    y = torch.linalg.vector_norm(X,ord = float('inf'), dim = 1)
    out = vectorized_function(y.detach().numpy())
    return torch.tensor(out, dtype = torch.float32)



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
output = triangle_tensor(x)
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
# Generate values for the function
#x_values = np.linspace(0, 0.99, 10000)
#num_points = 10000
#x = np.linspace(0, 1, num_points)
#y = np.linspace(0, 1, num_points)

# Create a meshgrid and reshape it into an array of shape (n, 2)
#X, Y = np.meshgrid(x, y)
#points = np.vstack([X.ravel(), Y.ravel()]).T
#print(points.shape)
#vectorized_function = np.vectorize(triangle_function)
#y_values = vectorized_function(np.linalg.norm(points, axis = 1))
#y_values = [triangle_function(x) for x in x_values]

# Plot the function
#plt.plot(x_values, y_values, label="Triangle Function")
#plt.xlabel("x")
#plt.ylabel("f(x)")
#plt.title("Piecewise Triangle Function")
#plt.grid(True)
#plt.show()