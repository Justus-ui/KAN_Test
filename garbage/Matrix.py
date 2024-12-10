import sympy as sp

# Define symbols for matrix elements
a0, b0, a1, b1 = sp.symbols('a0 b0 a1 b1')
a0p, b0p, a1p, b1p = sp.symbols('a0p b0p a1p b1p')

# Define the matrix M
M = sp.Matrix([
    [a0,  0,  0,  0, a0p,  0,  0,  0],
    [b0,  1,  0,  0, b0p,  1,  0,  0],
    [0,   0, a0,  0, a0p,  0,  0,  0],
    [0,   0, b0,  1, b0p,  1,  0,  0],
    [0,   0, a1,  0, a1p,  0,  0,  0],
    [0,   0, b1,  1, b1p,  1,  0,  0],
    [0,   0, a1,  0,   0,  0, a1p,  0],
    [0,   0, b1,  1,   0,  0, b1,   1]
])

A = sp.Matrix([
    [a0, a0p,0,0],
    [0,a0p,a0,0],
    [0,a1p,a1,0],
    [0,0,a1,a1p]
])

M = sp.Matrix([
    [a0,  0,  0,  0, a0p,  0,  0,  0],
    [b0,  1,  0,  0, b0p,  1,  0,  0],
    [0,   0, a0,  0, a0p,  0,  0,  0],
    [0,   0, b0,  1, b0p,  1,  0,  0],
    [0,   0, a1,  0, a1p,  0,  0,  0],
    [0,   0, b1,  1, b1p,  1,  0,  0],
    [0,   0, a1,  0,   0,  0, a1p,  0],
    [0,   0, b1,  1,   0,  0, b1,   1]
])
# Compute the determinant of M to check for invertibility
det_M = M.det()
A.inv() * sp.Matrix([])
print(det_M)
