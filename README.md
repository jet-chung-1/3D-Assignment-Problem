# 3D Assignment Problem

This repository provides a Python implementation of a dual-primal method for solving the 3D assignment problem.

## Functionality

The solver applies a Lagrangian relaxation technique to relax the 3D assignment problem to a 2D assignment problem, providing dual bounds. We use a modified subgradient method with Polyak step sizes and Nesterov acceleration to obtain tight dual bounds. We then reconstruct a locally optimal solution using a local search method inspired from 2-OPT. The Jupyter notebooks in this repository demonstrate additional functionality and usage examples.

### Usage
If we just want a solution using the default parameters, we can use the following code on a specified cost matrix C: 
```python
solver = Solver()
_, _, best_sol, best_value = solver.optimize(C) 
```

We can also set parameters for the solver; the ones displayed below are the defaults, and were found to give good performance across many problem sizes and instances. 
```python
learning_rate_scale="1/k"
learning_rate=0.1
max_iterations=100
threshold=0.01

solver = Solver(max_iterations, threshold, learning_rate, learning_rate_scale, beta) # all are optional 
dual_bounds, primal_bounds, best_sol, best_value = solver.optimize(C) 
```

### Subgradient Method
We use a modified Polyak step size with Nesterov acceleration for the subgradient method, which gave the strongest results for finding the Lagrangian dual quickly.

$$x^{k+1} = x^k - \alpha_k \partial f(x^k + \beta(x^k - x^{k-1})) + \beta(x^k - x^{k-1})
$$
where $$\alpha_k = \frac{\lambda((\text{dual value})_k - (\text{best value}))}{k\|\partial f(x^k + \beta(x^k - x^{k-1})\|_2^2}$$
where we took the parameter $\beta = 0.95$
and $\lambda$ is a hyperparameter which was found to give good results with $\lambda = 0.1$.


### Local Search
We first solve the problem with the greedy method, then perform a local search algorithm inspired from the 2-opt algorithm for TSP which finds very good primal solutions. Empirically we have found that the local search gives very good solutions in very fast times: for medium sized problems, we typically solve it within 1%, and for large problems, we often solve it within 0.1%.


## Solver Speed Benchmarking
We are able to solve large problems with up to $500 \times 500 \times 500$ cost matrices to very high accuracy within a few minutes. See the large_problems notebook and below.

```
--------------------------------------------------
FINISHED OPTIMIZATION
--------------------------------------------------
SEARCH VALUES
Greedy value: 49868.63685207059
Greedy + 2OPT value: 49982.23668548891
Best value: 49982.23668548891
--------------------------------------------------
OPTIMIZATION RESULTS
Primal: 49982.24 <= Value: 49982.24 <= Dual: 49999.66, duality %: 0.034860%, feasiblity: True
--------------------------------------------------

  _     ._   __/__   _ _  _  _ _/_   Recorded: 03:54:56  Samples:  1049
 /_//_/// /_\ / //_// / //_'/ //     Duration: 65.383    CPU time: 55.513
/   _/                      v4.6.2

Program: /Users/jetchung/opt/anaconda3/lib/python3.9/site-packages/ipykernel_launcher.py -f /Users/jetchung/Library/Jupyter/runtime/kernel-50ea3a89-6987-441d-a3bd-9689dbc48d8a.json

65.381 ZMQInteractiveShell.run_ast_nodes  IPython/core/interactiveshell.py:3209
└─ 65.380 <cell line: 3>  ../../../../../../var/folders/jv/2k_pqb6j1455zljdchhw7tpc0000gn/T/ipykernel_87258/1998096011.py:3
   └─ 65.380 Solver.optimize  main_optimized.py:56
      ├─ 48.408 Solver.greedy  main_optimized.py:202
      │  ├─ 45.171 argmax  numpy/core/fromnumeric.py:1140
      │  │     [3 frames hidden]  numpy, <built-in>
      │  │        45.129 ndarray.argmax  <built-in>
      │  ├─ 2.252 [self]  main_optimized.py
      │  └─ 0.831 ndarray.copy  <built-in>
      ├─ 15.831 Solver.local_process  main_optimized.py:222
      │  └─ 15.503 [self]  main_optimized.py
      └─ 0.995 Solver.objective_func  main_optimized.py:39
         └─ 0.813 [self]  main_optimized.py
```
## Solver Comparison

- The solver has been benchmarked against Gurobi and PuLP. Across various problem sizes and instances, our solver consistently demonstrates a significant speed-up.
- The solver halts within 50 iterations or when it approaches within 5% of optimal.
- For the tests, cost matrix entries are uniformly chosen from the range [0, 100], and all solvers run on the same problems for a given benchmarking instance.

A benchmarking utility is provided to facilitate easy comparison between different solvers. Here is an example:
```python
N = 30
num_problems = 10

solver1 = Solver(max_iterations=50, threshold=0.05, verbosity = False)
solver2 = GurobiSolver('/Users/jetchung/gurobi.json')

solvers = [solver1, solver2]
problems = create_problems(N, num_problems, verbosity = True)

df = benchmark(problems, solvers, verbosity = True)
```

```
--------------------------------------------------
Created 10 problem instances for size 30 with scale=100.
--------------------------------------------------


--------------------------------------------------
Benchmarking CustomSolver_1 Solver:
--------------------------------------------------
Instance 1: Objective Value: 2965.19, Duality % Gap: 0.97%
Instance 2: Objective Value: 2959.59, Duality % Gap: 1.11%
Instance 3: Objective Value: 2968.97, Duality % Gap: 0.87%
Instance 4: Objective Value: 2955.92, Duality % Gap: 1.20%
Instance 5: Objective Value: 2942.00, Duality % Gap: 1.73%
Instance 6: Objective Value: 2941.98, Duality % Gap: 1.69%
Instance 7: Objective Value: 2949.55, Duality % Gap: 1.40%
Instance 8: Objective Value: 2952.12, Duality % Gap: 1.38%
Instance 9: Objective Value: 2966.05, Duality % Gap: 0.93%
Instance 10: Objective Value: 2957.90, Duality % Gap: 1.19%
--------------------------------------------------
Avg. execution time for CustomSolver_1: 0.0760 seconds
--------------------------------------------------


--------------------------------------------------
Benchmarking GurobiSolver_2 Solver:
--------------------------------------------------
Instance 1: Objective Value: 2989.83
Instance 2: Objective Value: 2990.84
Instance 3: Objective Value: 2990.92
Instance 4: Objective Value: 2989.70
Instance 5: Objective Value: 2991.24
Instance 6: Objective Value: 2989.67
Instance 7: Objective Value: 2989.36
Instance 8: Objective Value: 2990.74
Instance 9: Objective Value: 2988.39
Instance 10: Objective Value: 2990.91
--------------------------------------------------
Avg. execution time for GurobiSolver_2: 4.5903 seconds
--------------------------------------------------
```
