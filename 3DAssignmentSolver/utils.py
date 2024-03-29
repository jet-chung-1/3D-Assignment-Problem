import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from main_optimized import Solver

def create_problems(N, num_problems, scale=100, verbosity=True):
    """
    Create random problem instances.

    Parameters:
        N: int
            Size of the problem.
        num_problems: int
            Number of problem instances to create.
        scale: int or float, optional
            Scale parameter for the beta distribution. Default is 100.
        beta: tuple, optional
            Shape parameters for the beta distribution. Default is (1, 1).
        verbosity: bool, optional
            Whether to print information during execution. Default is True.

    Returns:
        problems: list
            List of problem instances.
    """
    problems = [np.random.uniform(0, scale, size=(N, N, N)) for _ in range(num_problems)]

    if verbosity:
        print("\n")
        print(f"{'-' * 50}")
        print(f"Created {num_problems} problem instances for size {N} with scale={scale}.")
        print(f"{'-' * 50}")
        print("\n")

    return problems


def benchmark(problems, solvers, verbosity=True):
    """
    Benchmark different solvers.

    Parameters:
        problems: list
            List of problem instances.
        solvers: list
            List of solver instances.
        verbosity: bool, optional
            Whether to print information during execution. Default is True.

    Returns:
        df: pandas DataFrame
            DataFrame containing solver names and their execution times.
    """
    solver_times = {}

    for solver_index, solver in enumerate(solvers, start=1):
        solver_name = type(solver).__name__ + f"_{solver_index}"  # Append an index to distinguish between instances

        if isinstance(solver, Solver):
            solver_name = "Custom" + solver_name
        solver_times[solver_name] = []

        if verbosity:
            print(f"{'-' * 50}")
            print(f"Benchmarking {solver_name} Solver:")
            print(f"{'-' * 50}")

        fractions = []  # List to store fraction values for custom solvers

        for problem_index, problem in enumerate(problems, start=1):
            start_time = timeit.default_timer()
            if isinstance(solver, Solver):
                dual_bounds, primal_bounds, _, _,  = solver.optimize(problem)
                fraction = (dual_bounds[-1] - primal_bounds[-1])/primal_bounds[-1]
                fractions.append(fraction)
                if verbosity:
                    print(f"Instance {problem_index}: Objective Value: {primal_bounds[-1]:.2f}, Duality % Gap: {100 * fraction:.2f}%")
            else:
                primal, _ = solver.optimize(problem)
                if verbosity:
                    print(f"Instance {problem_index}: Objective Value: {primal:.2f}")
            elapsed_time = timeit.default_timer() - start_time

            solver_times[solver_name].append(elapsed_time)

        avg_time = np.mean(solver_times[solver_name])
        if isinstance(solver, Solver):
            if verbosity:
                print(f"{'-' * 50}")
                print(f"Avg. execution time for {solver_name}: {avg_time:.4f} seconds")
                print(f"{'-' * 50}")
        else:
            if verbosity:
                print(f"{'-' * 50}")
                print(f"Avg. execution time for {solver_name}: {avg_time:.4f} seconds")
                print(f"{'-' * 50}")
        print("\n")

    df = pd.DataFrame.from_dict(solver_times, orient='index').transpose()
    df.index = range(1, len(df) + 1)
    return df

def duality_visualizer(dual_bounds, primal_bounds, figsize=(8, 6)):
    """
    Visualizes the primal and dual bounds during optimization.

    Parameters:
        dual_bounds (list): List of dual objective values.
        primal_bounds (list): List of primal objective values.
        figsize (tuple, optional): Figure size. Default is (8, 6).
    """
    fraction = (dual_bounds[-1] - primal_bounds[-1]) / primal_bounds[-1]
    halfway_point = (dual_bounds[-1] + primal_bounds[-1]) / 2

    plt.figure(figsize=figsize)

    colors = sns.color_palette("Set2", 2)

    plt.plot(dual_bounds, label='Dual Bound', color=colors[0], alpha=0.7)
    plt.plot(primal_bounds, label='Primal Bound', color=colors[1], alpha=0.7)

    plt.axhline(y=halfway_point, color='gray', linestyle='--', label='Halfway Point')

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.title(f'Primal and Dual Bounds, within {fraction * 100:.2f}% of optimal')

    plt.legend()
    plt.show()


def is_feasible(x):
    """
    Check if a given solution is feasible.

    Parameters:
        x: numpy array
            Primal solution to be checked.

    Returns:
        bool: True if feasible, False otherwise.
    """
    if not np.all(np.sum(x, axis=(1, 2)) == 1):
        return False

    if not np.all(np.sum(x, axis=(0, 2)) == 1):
        return False

    if not np.all(np.sum(x, axis=(0, 1)) == 1):
        return False
    
    return True

def value(x, C):
    """
    Compute the value of the objective function for a given primal solution.

    Parameters:
        x: numpy array
            Primal solution.
        C: numpy array
            Cost matrix.

    Returns:
        float: Objective function value.
    """
    return np.sum(C * x)
