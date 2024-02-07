import numpy as np
from scipy.optimize import linear_sum_assignment


class Solver:
    def __init__(self, verbosity = False):
        self.verbosity = verbosity # will add some more output later

    def objective_func(self, u):
        """
        Objective function for optimization.

        Parameters:
            u: numpy array
                Lagrange multiplier.

        Returns:
            dual_value: float
                Dual objective value.
            x: numpy array
                Constructed primal solution.
        """
        # Compute the sum of C and u
        u = u.reshape(1, 1, -1)
        sum_C_u = self.C + u

        # Find indices of maximum elements
        max_indices = np.argmax(sum_C_u, axis=-1)
        D = sum_C_u[np.arange(self.N)[:, None], np.arange(self.N), max_indices]
        K = max_indices.squeeze()

        # Solve linear assignment problem in 2D (worth noting this is fast)
        rows, cols = linear_sum_assignment(-D)

        # Construct primal solution
        x = np.zeros((self.N, self.N, self.N), dtype=int)
        x[rows, cols, K[rows, cols]] = 1

        return np.sum(sum_C_u * x) - np.sum(u), x

    def construct(self, x):
        """
        Construct primal solution.

        Parameters:
            x: numpy array
                Solution of relaxation.

        Returns:
            y: numpy array
                Constructed primal solution.
        """
        # Initialize arrays
        J = np.zeros(self.N, dtype=int)
        D = np.zeros((self.N, self.N), dtype=int)

        # Construct primal solution
        for i in range(self.N):
            j_indices, k_indices = np.where(x[i] == 1)

            if j_indices.size > 0:
                j = j_indices[0]
                k = k_indices[0]
                J[i] = j
                D[i, k] = self.C[i, j, k]

        # Solve linear assignment problem in 2D (worth noting this is fast)
        k_indices = linear_sum_assignment(-D)[1]

        y = np.zeros((self.N, self.N, self.N), dtype=int)
        y[np.arange(self.N), J, k_indices] = 1

        return y

    def subgrad_func(self, u):
        """
        Compute subgradient.

        Parameters:
            u: numpy array
                Point in the dual space.

        Returns:
            subgradient: numpy array
                Subgradient.
        """
        # Compute objective function and primal solution
        _, x = self.objective_func(u)
        # Compute sum along the first dimension
        sum_along_first_dim = np.sum(x, axis=(0, 1))
        # Compute subgradient
        subgradient = sum_along_first_dim - 1
        return subgradient

    def compute_learning_rate(self, k):
        """
        Compute learning rate.

        Parameters:
            k: int
                Iteration number.

        Returns:
            learning_rate: float
                Learning rate.
        """
        if self.learning_rate_scale == "constant":
            return self.learning_rate
        elif self.learning_rate_scale == "1/k":
            return self.learning_rate / (k + 1)
        elif self.learning_rate_scale == "1/sqrt(k)":
            return self.learning_rate / np.sqrt(k + 1)
        else:
            raise ValueError("Invalid learning rate scale specified.")

    def subgradient_algorithm(self, initial_point):
        """
        Subgradient algorithm.

        Parameters:
            initial_point: numpy array
                Initial point in the dual space.

        Returns:
            dual_bounds: list
                Dual objective values.
            primal_bounds: list
                Primal objective values.
            best_sol: numpy array
                Best primal solution found.
            best_value: float
                Best primal objective value found.
            delta: float
                Difference between dual and primal objective values.
            fraction: float
                Fraction of delta over best_value.
        """
        # Initialize variables
        u = np.array(initial_point)
        k = 0

        # Compute initial dual objective value and primal solution
        dual_value, x_point = self.objective_func(u)
        best_sol = self.construct(x_point)
        best_value = self.value(best_sol)

        dual_bounds = [dual_value]
        primal_bounds = [best_value]

        delta = dual_value - best_value

        fraction = delta / best_value

        # Main optimization loop
        while fraction > self.threshold and k < self.max_iterations:
            # Compute dual objective value and primal solution
            dual_value, x_point = self.objective_func(u)
            primal_construction = self.construct(x_point)
            primal_value = self.value(primal_construction)

            # Update best primal solution and value
            if primal_value > best_value:
                best_value = primal_value
                best_sol = primal_construction

            # Update delta and fraction
            delta = dual_value - best_value
            fraction = delta / best_value

            dual_bounds.append(dual_value)
            primal_bounds.append(best_value)

            # Compute subgradient
            subgradient = self.subgrad_func(u)

            # Update u
            u = u - self.compute_learning_rate(k) * subgradient

            k += 1

        return dual_bounds, primal_bounds, best_sol, best_value, delta, fraction 

    def nesterov_accelerated_gradient(self, initial_point):
        """
        Nesterov accelerated gradient algorithm.

        Parameters:
            initial_point: numpy array
                Initial point in the dual space.

        Returns:
            dual_bounds: list
                Dual objective values.
            primal_bounds: list
                Primal objective values.
            best_sol: numpy array
                Best primal solution found.
            best_value: float
                Best primal objective value found.
            delta: float
                Difference between dual and primal objective values.
            fraction: float
                Fraction of delta over best_value.
        """
        # Initialize variables
        u = np.array(initial_point)
        y = np.array(initial_point)

        k = 0

        # Compute initial dual objective value and primal solution
        dual_value, x_point = self.objective_func(u)
        best_sol = self.construct(x_point)
        best_value = self.value(best_sol)

        dual_bounds = [dual_value]
        primal_bounds = [best_value]

        delta = dual_value - best_value

        fraction = delta / best_value

        beta = self.beta

        # Main optimization loop
        while fraction > self.threshold and k < self.max_iterations:
            # Compute dual objective value and primal solution
            dual_value, x_point = self.objective_func(y)
            primal_construction = self.construct(x_point)
            primal_value = self.value(primal_construction)

            # Update best primal solution and value
            if primal_value > best_value:
                best_value = primal_value
                best_sol = primal_construction

            # Update delta and fraction
            delta = dual_value - best_value
            fraction = delta / best_value

            dual_bounds.append(dual_value)
            primal_bounds.append(best_value)

            # Compute subgradient
            subgradient = self.subgrad_func(y)

            # Polyak's Step Size Rule
            numerator = dual_value - best_value
            denominator = np.linalg.norm(subgradient) ** 2
            learning_speed = (
                self.compute_learning_rate(k) * numerator / denominator
                if denominator != 0
                else self.compute_learning_rate(k)
            )

            # Update y and u
            y = u - learning_speed * subgradient
            u = y + beta * (y - u)

            k += 1

        return dual_bounds, primal_bounds, best_sol, best_value, delta, fraction 


    def is_feasible(self, x):
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
    
    def value(self, x):
        """
        Compute the value of the objective function for a given primal solution.

        Parameters:
            x: numpy array
                Primal solution.

        Returns:
            float: Objective function value.
        """
        return np.sum(self.C * x)
    

    def optimize(
        self,
        C,
        learning_rate_scale="constant",
        algorithm="nesterov",
        beta=0.95,
        search_size=10,
        learning_rate=0.1,
        max_iterations=200,
        threshold=0.05,
    ):
        self.C = C
        self.learning_rate_scale = learning_rate_scale
        self.beta = beta
        self.search_size = search_size
        self.algorithm = algorithm
        self.N = C.shape[0]
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.threshold = threshold

        # Check if algorithm is valid
        allowed_algorithms = ["subgradient", "nesterov"]
        if self.algorithm not in allowed_algorithms:
            raise ValueError(
                f"Invalid algorithm '{self.algorithm}'. Please choose from: {allowed_algorithms}"
            )
        """
        Optimize the objective function using the selected algorithm.

         Parameters:
            C: numpy array
                Cost matrix.
            learning_rate_scale: str, optional
                Scale factor for learning rate. Default is "constant".
            algorithm: str, optional
                Optimization algorithm to use. Default is "nesterov".
            beta: float, optional
                Beta parameter for Nesterov's accelerated gradient descent. Default is 0.95.
            search_size: int, optional
                Search size for optimization. Default is 10.
            learning_rate: float, optional
                Initial learning rate. Default is 0.1.
            max_iterations: int, optional
                Maximum number of iterations. Default is 200.
            threshold: float, optional
                Convergence threshold. Default is 0.05.

        Returns:
            dual_bounds: list
                Dual objective values.
            primal_bounds: list
                Primal objective values.
            best_sol: numpy array
                Best primal solution found.
            best_value: float
                Best primal objective value found.
            delta: float
                Difference between dual and primal objective values.
            fraction: float
                Fraction of delta over best_value.
        """

        initial_point = np.random.uniform(-self.search_size, self.search_size, self.N)
        if self.algorithm == "subgradient":                
            return self.subgradient_algorithm(initial_point)
        elif self.algorithm == "nesterov":
            return self.nesterov_accelerated_gradient(initial_point)
        else:
            raise ValueError(f"Invalid algorithm '{self.algorithm}'. Please choose from: ['subgradient', 'nesterov']")
        
