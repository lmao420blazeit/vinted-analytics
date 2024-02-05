from scipy.optimize import linprog

# Coefficients of the objective function to minimize (since linprog minimizes)
c = [-8, -11, -9]

# Coefficients of the inequality constraints
A = [
    [-0.2, -0.3, -0.24],  # Coefficients for the first constraint: 0.2*mice + 0.3*keyboards + 0.24*joysticks > 13000
    [-0.04, -0.55, -0.04],  # Coefficients for the second constraint: 0.04*mice + 0.55*keyboards + 0.04*joysticks > 1500
    [1, 0, 0],  # Coefficients for the third constraint: mice < 15000
    [0, 1, 0],  # Coefficients for the fourth constraint: keyboards < 29000
    [0, 0, 1]   # Coefficients for the fifth constraint: joysticks < 11000
]

# Right-hand side values of the inequality constraints
b = [13000, 1500, 15000, 29000, 11000]

# Bounds for variables (non-negativity constraints)
mice_bounds = (0, None)
keyboards_bounds = (0, None)
joysticks_bounds = (0, None)

# Solve the linear programming problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[mice_bounds, keyboards_bounds, joysticks_bounds], method='highs')

# Display the results
print("Optimal values:")
print("mice =", result.x[0])
print("keyboards =", result.x[1])
print("joysticks =", result.x[2])
print("Minimized cost:", result.fun)  # Convert back to maximize
