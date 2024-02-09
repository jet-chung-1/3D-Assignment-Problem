import numpy as np
import timeit
import pandas as pd

from main import Solver
from gurobi_solver import GurobiSolver
from pulp_solver import PulpSolver

def create_problems(N, num_problems, verbosity=True):
    """
    Create random problem instances.

    Parameters:
        N: int
            Size of the problem.
        num_problems: int
            Number of problem instances to create.
        verbosity: bool, optional
            Whether to print information during execution. Default is True.

    Returns:
        problems: list
            List of problem instances.
    """
    problems = []
    for i in range(num_problems):
        C = np.random.beta(1, 1, size=(N, N, N))*100  # Cost matrices are uniform in [0, 100]
        problems.append(C)
        # if verbosity:
        #     print(f"{i}: {C[0,0,0]}")

    if verbosity:
        print(f"Created {num_problems} problem instances for size {N}.")
        print("-" * 10)

    return problems


def benchmark(problems, solvers, verbosity=True):
    """
    Benchmark different solvers.

    Parameters:
        N: int
            Size of the problem.
        num_problems: int
            Number of problem instances to create.
        verbosity: bool, optional
            Whether to print information during execution. Default is True.
    """
    solver_times = {}
     
    for solver_index, solver in enumerate(solvers, start=1):
        solver_name = type(solver).__name__ + f"_{solver_index}"  # Append an index to distinguish between instances

        if isinstance(solver, Solver):
            solver_name = "Custom" + solver_name
        solver_times[solver_name] = []

        if verbosity:
            print(f"Benchmarking {solver_name} Solver:")

        if isinstance(solver, Solver):
            fraction_below_threshold = 0
            for problem in problems:
                start_time = timeit.default_timer()
                _, primal_bounds, _, _, _, fraction = solver.optimize(problem)
                if verbosity:
                    print(f"Objective Value: {primal_bounds[-1]:.2f}, Duality % Gap: {100 * fraction:.2f}%")
                elapsed_time = timeit.default_timer() - start_time

                solver_times[solver_name].append(elapsed_time)

                if fraction < solver._threshold:
                    fraction_below_threshold += 1

                # cumulative_times = np.cumsum(solver_times[type(solver).__name__])
                avg_time = np.mean(solver_times[solver_name])
                percentage_below_threshold = (fraction_below_threshold / len(problems)) * 100
            
            if verbosity:
                print(f"Avg. execution time for {solver_name}: {avg_time:.4f} seconds")
                print(f"Percentage of time fraction < {solver._threshold * 100:.2f}%: {percentage_below_threshold:.2f}%")
                print("-" * 10)

        elif isinstance(solver, GurobiSolver) or isinstance(solver, PulpSolver):

            for problem in problems:
                start_time = timeit.default_timer()
                primal = solver.optimize(problem)[0]
                if verbosity:
                    print(f"Objective Value: {primal:.2f}")
                elapsed_time = timeit.default_timer() - start_time

                solver_times[solver_name].append(elapsed_time)
                avg_time = np.mean(solver_times[solver_name])
            if verbosity:
                print(f"Avg. execution time for {solver_name}: {avg_time:.4f} seconds")
                print("-" * 10)
       
        else:
            print("Error: solver class not known")

    df = pd.DataFrame.from_dict(solver_times, orient='index').transpose()
    df.index = range(1, len(df) + 1)
    return df
    

