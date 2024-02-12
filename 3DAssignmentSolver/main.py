import numpy as np
from scipy.optimize import linear_sum_assignment

class Solver:
    """
    Solver class for 3D assignment problem using dual-primal method.
    """

    def __init__(
        self,
        learning_rate_scale="1/k",
        algorithm="nesterov",
        beta=0.95,
        search_size=0.01,
        learning_rate=0.1,
        max_iterations=200,
        threshold=0.05,
        verbosity=False,
        local_search=False,
        local_search_all=False,
        op_solver = True,
    ):
        """
        Initialize the Solver object with specified parameters.

        Parameters:
            learning_rate_scale (str, optional): Scale factor for learning rate. Default is "constant".
            algorithm (str, optional): Optimization algorithm to use. Default is "nesterov".
            beta (float, optional): Beta parameter for Nesterov's accelerated gradient descent. Default is 0.95.
            search_size (int, optional): Search size for optimization. Default is 10.
            learning_rate (float, optional): Initial learning rate. Default is 0.1.
            max_iterations (int, optional): Maximum number of iterations. Default is 200.
            threshold (float, optional): Convergence threshold. Default is 0.05.
            verbosity (bool, optional): Whether to print additional information during execution. Default is False.
        """
        self._learning_rate_scale = learning_rate_scale
        self._beta = beta
        self._search_size = search_size
        self._algorithm = algorithm
        self._learning_rate = learning_rate
        self._max_iterations = max_iterations
        self._threshold = threshold
        self._verbosity = verbosity
        self._local_search = local_search
        self._local_search_all = local_search_all
        self._op_solver = op_solver

    @property
    def threshold(self):
        """
        Get the convergence threshold.

        Returns:
            float: Convergence threshold.
        """
        return self._threshold
    
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


        if self._local_search_all:
            y_local = self.local_constructor(x)
            if y_local is not None and self.value(y_local) > self.value(y):
                return y_local
        else:
            return y
    
    def local_constructor(self, x):
        ct = 0 
        while ct < self.N:
            S = np.sum(x, axis=(0, 1))
            A = np.where(S == 0)[0]
            B = np.where(S > 1)[0]
            E = np.where(S == 1)[0]
    #         print(f" < 1: {A}, > 1: {B}, = 1 {E}")
            if len(A) == 0:
                break 
            else:
                D = np.full((self.N, self.N), -np.inf)
                argmax_diff = np.full((self.N, self.N, 2), -np.inf)  # 3D array to store indices (i, j, k, l)
                for i in range(self.N):
                    for j in range(self.N):
                        A = np.where((x[i, j, :] == 0) & (S == 0))[0] # Get the indices where both x and S are 0
                        B = np.where((x[i, j, :] > 0) & (S > 1))[0]   # Get the indices where x > 0 and S > 1
                        if len(A) > 0 and len(B) > 0:
                            max_diff = -np.inf
                            for k in A:
                                for l in B:
                                    diff = self.C[i, j, k] - self.C[i, j, l]
                                    if diff > max_diff:
                                        max_diff = diff
                                        argmax_diff[i, j] = [k, l]  # Store the indices of the maximizing values
                            D[i, j] = max_diff

                i, j = np.unravel_index(np.argmax(D), (self.N,self.N))
                k, l = argmax_diff[i,j]
                x[i, j, int(k)] = 1
                x[i, j, int(l)] = 0

    #             print(f"swap {i,j,k} <- 1, {i, j, l} <- 0")
    #             print(f"change by {C[i,j,int(k)] - C[i,j,int(l)]}" )
                
    #             print("-"*50)
                
                ct += 1
        
        if self.is_feasible(x):
            print("hi")
            print(self.value(x))
            x = self.local_process(x)
            print(self.value(x))
            return x
        else:
            return None
            
    # still need to numpy this all    
    def local_construct(self):
    
        # just initial bound
        u = np.zeros(self.N)
        u = u.reshape(1, 1, -1)
        sum_C_u = self.C + u
        
        max_indices = np.argmax(sum_C_u, axis=-1)
        D = sum_C_u[np.arange(self.N)[:, None], np.arange(self.N), max_indices]
        K = max_indices.squeeze()

        rows, cols = linear_sum_assignment(-D)

        x = np.zeros((self.N, self.N, self.N), dtype=int)
        x[rows, cols, K[rows, cols]] = 1

    
        sol = self.local_constructor(x)
        val = self.value(sol)
        
        return sol, val
        
  

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
        if self._learning_rate_scale == "constant":
            return self._learning_rate
        elif self._learning_rate_scale == "1/k":
            return self._learning_rate / (k + 1)
        elif self._learning_rate_scale == "1/sqrt(k)":
            return self._learning_rate / np.sqrt(k + 1)
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
        """
        # Initialize variables
        u = np.array(initial_point)
        k = 0

        # Compute initial dual objective value and primal solution
        dual_value, x_point = self.objective_func(u)

        best_sol = self.construct(x_point)
        best_value = self.value(best_sol)

        # print("best-val: ", best_value)
        if self._local_search:
            sol, val = self.local_construct()
            # print("local-val: ", local_construct_val)
            if val is not None and val > best_value:
                best_value = local_construct_val
                best_sol = sol

        dual_bounds = [dual_value]
        primal_bounds = [best_value]

        fraction = (dual_value - best_value) / best_value
        
        # Main optimization loop
        while fraction > self._threshold and k < self._max_iterations:
            # Compute dual objective value and primal solution
            dual_value, x_point = self.objective_func(u)
            primal_construction = self.construct(x_point)
            primal_value = self.value(primal_construction)

            # Update best primal solution and value
            if primal_value > best_value:
                best_value = primal_value
                best_sol = primal_construction

            # Update fraction
            fraction = (dual_value - best_value) / best_value

            dual_bounds.append(dual_value)
            primal_bounds.append(best_value)

            # Compute subgradient
            subgradient = self.subgrad_func(u)

            # Update u
            u = u - self.compute_learning_rate(k) * subgradient

            k += 1
                    
        print(self.is_feasible(best_sol), self.value(best_sol))
        best_sol = self.local_process(best_sol)
        best_value = self.value(best_sol)
        print(self.is_feasible(best_sol), self.value(best_sol))

        dual_bounds.append(dual_bounds[-1])
        primal_bounds.append(best_value)
        return dual_bounds, primal_bounds, best_sol, best_value

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
        """
        # Initialize variables
        u = np.array(initial_point)
        y = np.array(initial_point)

        k = 0

        # Compute initial dual objective value and primal solution
        dual_value, x_point = self.objective_func(u)

        best_sol = self.construct(x_point)
        best_value = self.value(best_sol)
 
        # print("best-val: ", best_value)
        if self._local_search:
            sol, val = self.local_construct()
            # print("local-val: ", local_construct_val)
            if val is not None and val > best_value:
                best_value = val
                best_sol = sol

        dual_bounds = [dual_value]
        primal_bounds = [best_value]

        fraction = (dual_value - best_value) / best_value

        beta = self._beta

        # Main optimization loop
        while fraction > self._threshold and k < self._max_iterations:
            # Compute dual objective value and primal solution
            dual_value, x_point = self.objective_func(y)
            primal_construction = self.construct(x_point)
            primal_value = self.value(primal_construction)

            # Update best primal solution and value

            if primal_value > best_value:
                best_value = primal_value
                best_sol = primal_construction

            # Update and fraction
            fraction = (dual_value - best_value) / best_value

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

        print(self.is_feasible(best_sol), self.value(best_sol))
        best_sol = self.local_process(best_sol)
        best_value = self.value(best_sol)
        print(self.is_feasible(best_sol), self.value(best_sol))

        dual_bounds.append(dual_bounds[-1])
        primal_bounds.append(best_value)
        return dual_bounds, primal_bounds, best_sol, best_value


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

    def local_process(self, x):
        if not self._op_solver:
            return x
        I = np.zeros((self.N, self.N))
        J = np.zeros((self.N, self.N))
        K = np.zeros((self.N, self.N))

        while True:

            for i1 in range(self.N):
                for i2 in range(self.N):
                    j1, k1 = np.where(x[i1,:,:])
                    j2, k2 = np.where(x[i2,:,:])
                    I[i1, i2] = self.C[i2, j1, k1].sum() + self.C[i1, j2, k2].sum() - self.C[i1, j1, k1].sum() - self.C[i2, j2, k2].sum()


            for j1 in range(self.N):
                for j2 in range(self.N):
                    i1, k1 = np.where(x[:,j1,:])
                    i2, k2 = np.where(x[:,j2,:])
                    J[j1, j2] = self.C[i1, j2, k1].sum() + self.C[i2, j1, k2].sum() - self.C[i1, j1, k1].sum() - self.C[i2, j2, k2].sum()

            for k1 in range(self.N):
                for k2 in range(self.N):
                    i1, j1 = np.where(x[:,:,k1])
                    i2, j2 = np.where(x[:,:,k2])
                    K[k1, k2] = self.C[i1, j1, k2].sum() + self.C[i2, j2, k1].sum() - self.C[i1, j1, k1].sum() - self.C[i2, j2, k2].sum()

            max_I_value = np.max(I)

            max_I_indices = np.unravel_index(np.argmax(I), I.shape)
            max_J_value = np.max(J)

            max_J_indices = np.unravel_index(np.argmax(J), J.shape)
            max_K_value = np.max(K)

            max_K_indices = np.unravel_index(np.argmax(K), K.shape)


            if max_I_value <= 0 and max_J_value <= 0 and max_K_value <= 0:
                break

            if max_I_value >= max_J_value and max_I_value >= max_K_value:
                if max_I_value > 0:
                    i1, i2 = max_I_indices
                    j1, k1 = np.where(x[i1, :, :])
                    j2, k2 = np.where(x[i2, :, :])

                    x[i1, j1, k1] = 0
                    x[i2, j2, k2] = 0
                    x[i2, j1, k1] = 1
                    x[i1, j2, k2] = 1

            elif max_J_value >= max_I_value and max_J_value >= max_K_value:
                if max_J_value > 0:
                    j1, j2 = max_J_indices
                    i1, k1 = np.where(x[:, j1, :])
                    i2, k2 = np.where(x[:, j2, :])

                    x[i1, j1, k1] = 0
                    x[i2, j2, k2] = 0
                    x[i1, j2, k1] = 1
                    x[i2, j1, k2] = 1  

            else:
                if max_K_value > 0:
                    k1, k2 = max_K_indices
                    i1, j1 = np.where(x[:, :, k1])
                    i2, j2 = np.where(x[:, :, k2])

                    x[i1, j1, k1] = 0
                    x[i2, j2, k2] = 0
                    x[i1, j1, k2] = 1
                    x[i2, j2, k1] = 1

        return x
         

    # # def local_process(self, x):
    #     return x
    #     non_zero_indices = np.nonzero(x)

    #     I = np.zeros((N, N))
    #     for idx1 in range(N):
    #         for idx2 in range(N):
    #             i1, j1, k1 = non_zero_indices[0][idx1], non_zero_indices[1][idx1], non_zero_indices[2][idx1]
    #             i2, j2, k2 = non_zero_indices[0][idx2], non_zero_indices[1][idx2], non_zero_indices[2][idx2]
    #             I[idx1, idx2] = C[i2, j1, k1] + C[i1, j2, k2] - C[i1, j1, k1] - C[i2, j2, k2]

    #     J = np.zeros((N, N))
    #     for idx1 in range(N):
    #         for idx2 in range(N):
    #             i1, j1, k1 = non_zero_indices[0][idx1], non_zero_indices[1][idx1], non_zero_indices[2][idx1]
    #             i2, j2, k2 = non_zero_indices[0][idx2], non_zero_indices[1][idx2], non_zero_indices[2][idx2]
    #             J[idx1, idx2] = C[i1, j2, k1] + C[i2, j1, k2] - C[i1, j1, k1] - C[i2, j2, k2]

    #     K = np.zeros((N, N))
    #     for idx1 in range(N):
    #         for idx2 in range(N):
    #             i1, j1, k1 = non_zero_indices[0][idx1], non_zero_indices[1][idx1], non_zero_indices[2][idx1]
    #             i2, j2, k2 = non_zero_indices[0][idx2], non_zero_indices[1][idx2], non_zero_indices[2][idx2]
    #             K[idx1, idx2] = C[i1, j1, k2] + C[i2, j2, k1] - C[i1, j1, k1] - C[i2, j2, k2]
        
    #     while True:
    #         max_I_value = np.max(I)
    #         max_I_indices = np.unravel_index(np.argmax(I), I.shape)

    #         max_J_value = np.max(J)
    #         max_J_indices = np.unravel_index(np.argmax(J), J.shape)

    #         max_K_value = np.max(K)
    #         max_K_indices = np.unravel_index(np.argmax(K), K.shape)
            
    #         if max_I_value <= 0 and max_J_value <= 0 and max_K_value <= 0:
    #             break

    #         if max_I_value >= max_J_value and max_I_value >= max_K_value:
    #             if max_I_value > 0:

    #                 i1, i2 = max_I_indices
    #                 j1, k1 = np.where(x[i1, :, :])
    #                 j2, k2 = np.where(x[i2, :, :])

    #                 x[i1, j1, k1] = 0
    #                 x[i2, j2, k2] = 0
    #                 x[i2, j1, k1] = 1
    #                 x[i1, j2, k2] = 1
                    
    #                 J[j1, j2] -= max_I_value
    #                 J[j2, j1] -= max_I_value
                    
    #                 K[k1, k2] -= max_I_value
    #                 K[k2, k1] -= max_I_value
                    
    #                 I[i1, i2] = 0
    #                 I[i2, i1] = 0

    #         elif max_J_value >= max_I_value and max_J_value >= max_K_value:
    #             if max_J_value > 0:

    #                 j1, j2 = max_J_indices
    #                 i1, k1 = np.where(x[:, j1, :])
    #                 i2, k2 = np.where(x[:, j2, :])

    #                 x[i1, j1, k1] = 0
    #                 x[i2, j2, k2] = 0
    #                 x[i1, j2, k1] = 1
    #                 x[i2, j1, k2] = 1

    #                 I[i1, i2] -= max_J_value
    #                 I[i2, i1] -= max_J_value
                    
    #                 K[k1, k2] -= max_J_value
    #                 K[k2, k1] -= max_J_value
                    
    #                 J[j1, j2] = 0
    #                 J[j2, j1] = 0

    #         else:
    #             if max_K_value > 0:
                    
    #                 k1, k2 = max_K_indices    
    #                 i1, j1 = np.where(x[:, :, k1])
    #                 i2, j2 = np.where(x[:, :, k2])

    #                 x[i1, j1, k1] = 0
    #                 x[i2, j2, k2] = 0
    #                 x[i1, j1, k2] = 1
    #                 x[i2, j2, k1] = 1

    #                 I[i1, i2] -= max_K_value
    #                 I[i2, i1] -= max_K_value
                    
    #                 J[j1, j2] -= max_K_value
    #                 J[j2, j1] -= max_K_value
                    
    #                 K[k1, k2] = 0
    #                 K[k2, k1] = 0
            


    def optimize(self, C):
        """
        Optimize the objective function using the selected algorithm.

        Parameters:
            C (numpy.array): Cost matrix.

        Returns:
            dual_bounds (list): Dual objective values.
            primal_bounds (list): Primal objective values.
            best_sol (numpy.array): Best primal solution found.
            best_value (float): Best primal objective value found.
        """
        self.C = C
        self.N = C.shape[0]

        # Check if algorithm is valid
        allowed_algorithms = ["subgradient", "nesterov"]
        if self._algorithm not in allowed_algorithms:
            raise ValueError(
                f"Invalid algorithm '{self._algorithm}'. Please choose from: {allowed_algorithms}"
            )
  
        initial_point = np.random.uniform(-self._search_size, self._search_size, self.N)
        if self._algorithm == "subgradient":                
            return self.subgradient_algorithm(initial_point)
        elif self._algorithm == "nesterov":
            return self.nesterov_accelerated_gradient(initial_point)
        else:
            raise ValueError(f"Invalid algorithm '{self._algorithm}'. Please choose from: ['subgradient', 'nesterov']")
        
       