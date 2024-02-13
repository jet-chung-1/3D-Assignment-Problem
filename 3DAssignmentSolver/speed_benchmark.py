from sys import path
path.append('../3DAssignmentSolver')
import numpy as np
from utils import duality_visualizer
from main_optimized import Solver
import seaborn as sns

def is_feasible(x):
    if len(np.unique(x[0])) + len(np.unique(x[1])) + len(np.unique(x[2])) < 3*N:
        return 
    return True

N = 500

C = np.random.randint(0, 100, size=(N, N, N))

solver = Solver(max_iterations=100, threshold=0.05, verbosity = True, search_size=0.01)

d, p, x, v = solver.optimize(C)


print(f"Primal: {p[-1]:.2f} <= Value: {v:.2f} <= Dual: {d[-1]:.2f}, duality %: {100*(d[-1] - p[-1])/p[-1]:.2f} feasible: {is_feasible(x)}")
duality_visualizer(d, p)
