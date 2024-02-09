import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from main import Solver

def create_problems(N, num_problems, scale=100, beta=(1, 1), verbosity=True):
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
    problems = []
    for _ in range(num_problems):
        C = np.random.beta(*beta, size=(N, N, N)) * scale  # Cost matrices are beta in [0, scale]
        problems.append(C)

    if verbosity:
        print(f"{'-' * 50}")
        if beta == (1, 1):
            print(f"Created {num_problems} problem instances for size {N} with scale={scale}.")
            print("Using a uniform distribution (beta = (1, 1)).")
        else:
            print(f"Created {num_problems} problem instances for size {N} with scale={scale} and beta={beta}.")
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
                _, primal_bounds, _, _, _, fraction = solver.optimize(problem)
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
            percentage_below_threshold = (sum(fraction < solver._threshold for fraction in fractions) / len(problems)) * 100
            if verbosity:
                print(f"{'-' * 50}")
                print(f"Avg. execution time for {solver_name}: {avg_time:.4f} seconds")
                print(f"Percentage of time fraction < {solver._threshold * 100:.2f}%: {percentage_below_threshold:.2f}%")
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
