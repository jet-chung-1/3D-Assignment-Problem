import numpy as np
from scipy.optimize import linear_sum_assignment



class Tensor:
    def __init__(self, i, j, k):
        self._i = i
        self._j = j
        self._k = k
                    
    def __repr__(self):
        return f"{self._i}-{self._j}-{self._k}"

    @property
    def i(self):
        return self._i
    
    @i.setter
    def i(self, value):
        self._i = value
    
    @property
    def j(self):
        return self._j
    
    @j.setter
    def j(self, value):
        self._j = value
    
    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, value):
        self._k = value


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
        max_iterations=100,
        threshold=0.05,
        verbosity=True,

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

        self._algorithm = algorithm
        self._beta = beta


        self._learning_rate = learning_rate
        self._learning_rate_scale = learning_rate_scale

        self._max_iterations = max_iterations
        self._threshold = threshold        

        self._search_size = search_size
            

        self._verbosity = verbosity

        if self._verbosity:
            print("\n")
            print("-"*50)
            print("Initializing Solver with parameters:")
            print("-"*50)
            print(f"Learning Rate Scale: {self._learning_rate_scale}")
            print(f"Algorithm: {self._algorithm}")
            print(f"Beta: {self._beta}")
            print(f"Learning Rate: {self._learning_rate}")
            print(f"Learning Rate Scale: {self._learning_rate_scale}")
            print(f"Maximum Iterations: {self._max_iterations}")
            print(f"Stopping Threshold: {self._threshold}")
            print("-"*50)
            print("Finished initializing Solver.")
            print("-"*50)
            print("\n")
            
            
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

        x = np.zeros((self.N, self.N, self.N), dtype=int)
        x[np.arange(self.N), J, k_indices] = 1

        return x
    
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
    #             print(f"change by {C[i,j,int(k)] - self.C[i,j,int(l)]}" )
                
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
                    
        best_sol = self.local_process(best_sol)
        best_value = self.value(best_sol)

        dual_bounds.append(dual_bounds[-1])
        primal_bounds.append(best_value)
        return dual_bounds, primal_bounds, best_sol, best_value
    
    def nesterov_accelerated_gradient(self, initial_point):
        """
        Nesterov accelerated gradient algorithm.

        Parameters:
            initial_point: numpy array
                Initial point in the dual space.
            verbose: bool, optional
                If True, prints verbose information. Default is False.

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
        if self._verbosity:
            print("\n")
            print("Starting Nesterov accelerated gradient algorithm...")
            print("-"*50)
            # print(f"Initial point: {initial_point}")

        # Initialize variables
        u = np.array(initial_point)  # Initialize u with the provided initial point
        y = np.array(initial_point)  # Initialize y with the provided initial point

        k = 0  # Initialize iteration counter

        # Compute initial dual objective value and primal solution
        dual_value, x_point = self.objective_func(u)  # Compute dual objective value
        best_sol = self.construct(x_point)  # Get primal solution from dual point
        best_value = self.value(best_sol)  # Calculate primal objective value

        # # Perform local search if enabled
        # if self._local_search and self._verbosity:
        #     print("Performing local search...")
        # if self._local_search:
        #     sol, val = self.local_construct()  # Perform local search
        #     if val is not None and val > best_value:
        #         best_value = val
        #         best_sol = sol
        #         if self._verbosity:
        #             print("Local search improved primal objective value.")

        # Initialize lists to store objective values
        dual_bounds = [dual_value]  # Initialize with initial dual objective value
        primal_bounds = [best_value]  # Initialize with initial primal objective value

        # Compute fraction for termination condition
        fraction = (dual_value - best_value) / best_value

        # Get beta value for Nesterov's acceleration
        beta = self._beta

        # Main optimization loop
        while fraction > self._threshold and k < self._max_iterations:
            if self._verbosity:
                print("-"*20)
                print(f"Iteration {k + 1}:")
                print(f"Primal value: {best_value:.3f}, Dual value: {dual_value:.3f}, Current duality %: {100*fraction:.2f}%")
            # Compute dual objective value and primal solution
            dual_value, x_point = self.objective_func(y)
            primal_construction = self.construct(x_point)
            primal_value = self.value(primal_construction)

            # Update best primal solution and value
            if primal_value > best_value:
                if self._verbosity:
                    print(f"Best primal solution updated from {primal_value} to {best_value}.")
                best_value = primal_value
                best_sol = primal_construction
                

            # Update fraction for termination condition
            fraction = (dual_value - best_value) / best_value

            # Append objective values to the lists
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

            # Update y and u using Nesterov's accelerated gradient update
            y = u - learning_speed * subgradient
            u = y + beta * (y - u)

            k += 1  # Increment iteration counter

        if self._verbosity:
            print("-"*20)
            print(f"Nesterov algorithm finished with primal value {best_value:.3f}, dual value {dual_value:.3f} and duality % {100*fraction:.2f}%")
            print("-"*20)
            print("-"*50)

            print("\n")
            print("Performing local search on primal...")
            
        # Perform local processing on the best primal solution found
        
        best_sol = self.local_process(best_sol, dual_value)
        best_value = self.value(best_sol)
        if self._verbosity:
            fraction = (dual_value - best_value)/best_value
            print("\n")
            print(f"Local search optimal value: {best_value:.3f},  dual value: {dual_value:.3f} and duality %: {100*fraction:.2f}%")
            print("-"*20)
        # Append the last dual objective value to the list
        dual_bounds.append(dual_bounds[-1])
        # Append the final primal objective value to the list
        primal_bounds.append(best_value)
        if self._verbosity:
            print(f"Finished optimization")

        # Return the results
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

    def local_process(self, x, dual_value):

        
        def swap_I(i1, i2):
            t1 = i_dict[i1] 
            t2 = i_dict[i2]

            j1, j2, k1, k2 = t1.j, t2.j, t1.k, t2.k

            a = self.C[t1.i, t1.j, t1.k] + self.C[t2.i, t2.j, t2.k]

            # swap
            t1.i, t2.i = i2, i1
            i_dict[i1], i_dict[i2] = t2, t1

            b = self.C[t1.i, t1.j, t1.k] + self.C[t2.i, t2.j, t2.k]

            I[i1, i2] = a - b
            I[i2, i1] = a - b

            # update J
            for idx in range(N):
                # update J
                # look at (idx, j1) for i2
                s = j_dict[j1]
                t = j_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, t.j, s.k] + self.C[t.i, s.j, t.k])

                J[idx, j1] = b - a
                J[j1, idx] = b - a

                # look at (idx, j2) for i1
                s = j_dict[j2]
                t = j_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, t.j, s.k] + self.C[t.i, s.j, t.k])

                J[idx, j2] = b - a
                J[j2, idx] = b - a

                # update K
                # look at (idx, k1) for i2
                s = k_dict[k1]
                t = k_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, s.j, t.k] + self.C[t.i, t.j, s.k])

                K[idx, k1] = b - a
                K[k1, idx] = b - a

                # look at (idx, k2) for i1
                s = k_dict[k2]
                t = k_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, s.j, t.k] + self.C[t.i, t.j, s.k])

                K[idx, k2] = b - a
                K[k2, idx] = b - a

                # update I, but not for i1 and i2

                if i1 != idx and i2 != idx:
                    s = i_dict[i1]
                    t = i_dict[idx]

                    a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                    b = (self.C[t.i, s.j, s.k] + self.C[s.i, t.j, t.k])

                    I[i1, idx] = b - a
                    I[idx, i1] = b - a

                    s = i_dict[i2]
                    t = i_dict[idx]

                    a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                    b = (self.C[t.i, s.j, s.k] + self.C[s.i, t.j, t.k])

                    I[i2, idx] = b - a
                    I[idx, i2] = b - a

        def swap_J(j1, j2):
            t1 = j_dict[j1] 
            t2 = j_dict[j2]

            i1, i2, k1, k2 = t1.i, t2.i, t1.k, t2.k

            a = self.C[t1.i, t1.j, t1.k] + self.C[t2.i, t2.j, t2.k]

            # swap
            t1.j, t2.j = j2, j1
            j_dict[j1], j_dict[j2] = t2, t1

            b = self.C[t1.i, t1.j, t1.k] + self.C[t2.i, t2.j, t2.k]

            J[j1, j2] = a - b
            J[j2, j1] = a - b

            # update I
            for idx in range(N):
                # update I
                # look at (i1, idx) for j2
                s = i_dict[i1]
                t = i_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[t.i, s.j, s.k] + self.C[s.i, t.j, t.k])

                I[i1, idx] = b - a
                I[idx, i1] = b - a

                # look at (i2, idx) for j1
                s = i_dict[i2]
                t = i_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[t.i, s.j, s.k] + self.C[s.i, t.j, t.k])

                I[i2, idx] = b - a
                I[idx, i2] = b - a

                # update K
                # look at (idx, k1) for j2
                s = k_dict[k1]
                t = k_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, s.j, t.k] + self.C[t.i, t.j, s.k])

                K[idx, k1] = b - a
                K[k1, idx] = b - a

                # look at (idx, k2) for j1
                s = k_dict[k2]
                t = k_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, s.j, t.k] + self.C[t.i, t.j, s.k])

                K[idx, k2] = b - a
                K[k2, idx] = b - a

                # update J, but not for j1 and j2
                if j1 != idx and j2 != idx:
                    s = j_dict[j1]
                    t = j_dict[idx]

                    a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                    b = (self.C[s.i, t.j, s.k] + self.C[t.i, s.j, t.k])

                    J[j1, idx] = b - a
                    J[idx, j1] = b - a

                    s = j_dict[j2]
                    t = j_dict[idx]

                    a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                    b = (self.C[s.i, t.j, s.k] + self.C[t.i, s.j, t.k])

                    J[j2, idx] = b - a
                    J[idx, j2] = b - a


        def swap_K(k1, k2):
            t1 = k_dict[k1] 
            t2 = k_dict[k2]

            i1, i2, j1, j2 = t1.i, t2.i, t1.j, t2.j

            a = self.C[t1.i, t1.j, t1.k] + self.C[t2.i, t2.j, t2.k]

            t1.k = k2
            t2.k = k1
            k_dict[k1] = t2
            k_dict[k2] = t1

            b = self.C[t1.i, t1.j, t1.k] + self.C[t2.i, t2.j, t2.k]


            K[k1, k2] = a - b
            K[k2, k1] = a - b

            # update I
            for idx in range(N):
                # update I
                # look at (i1, idx) for k2
                s = i_dict[i1]
                t = i_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[t.i, s.j, s.k] + self.C[s.i, t.j, t.k])

                I[i1,idx] = b - a
                I[idx,i1] = b - a

                # look at (i2, idx) for k1
                s = i_dict[i2]
                t = i_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[t.i, s.j, s.k] + self.C[s.i, t.j, t.k])

                I[i2,idx] = b - a
                I[idx,i2] = b - a

                ### 

                #update J
                # look at (j1, idx) for k2
                s = j_dict[j1]
                t = j_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, t.j, s.k] + self.C[t.i, s.j, t.k])

                J[j1, idx] = b - a
                J[idx, j1] = b - a

                # look at (j2, idx) for k1
                s = j_dict[j2]
                t = j_dict[idx]

                a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                b = (self.C[s.i, t.j, s.k] + self.C[t.i, s.j, t.k])

                J[j2, idx] = b-a
                J[idx, j2] = b-a

                #update K
                #
                if k1 != idx and k2 != idx:
                    s = k_dict[k1]
                    t = k_dict[idx]

                    a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                    b = (self.C[s.i, s.j, t.k] + self.C[t.i, t.j, s.k])
                    K[k1, idx] = b - a
                    K[idx, k1] = b - a

                    s = k_dict[k2]
                    t = k_dict[idx]

                    a = (self.C[s.i, s.j, s.k] + self.C[t.i, t.j, t.k])
                    b = (self.C[s.i, s.j, t.k] + self.C[t.i, t.j, s.k])

                    K[k2, idx] = b - a
                    K[idx, k2] = b - a


        i_dict = {}
        j_dict = {}
        k_dict = {}

        tensors = []
        N = np.shape(x)[0]
        for l in range(N):
            i = np.where(x==1)[0][l]
            j = np.where(x==1)[1][l]
            k = np.where(x==1)[2][l]

            t = Tensor(i, j, k)
            tensors.append(t)
            i_dict[i] = t
            j_dict[j] = t
            k_dict[k] = t

        i_dict, j_dict, k_dict, tensors

        I = np.zeros((self.N, self.N))
        J = np.zeros((self.N, self.N))
        K = np.zeros((self.N, self.N))
        
        for idx1 in range(N):
            for idx2 in range(N):  
                t1 = i_dict[idx1]
                t2 = i_dict[idx2]
                I[idx1, idx2] = self.C[t2.i, t1.j, t1.k] + self.C[t1.i, t2.j, t2.k] - self.C[t1.i, t1.j, t1.k] - self.C[t2.i, t2.j, t2.k]

                t1 = j_dict[idx1]
                t2 = j_dict[idx2]
                J[idx1, idx2] = self.C[t1.i, t2.j, t1.k] + self.C[t2.i, t1.j, t2.k] - self.C[t1.i, t1.j, t1.k] - self.C[t2.i, t2.j, t2.k]

                t1 = k_dict[idx1]
                t2 = k_dict[idx2]
                K[idx1, idx2] = self.C[t1.i, t1.j, t2.k] + self.C[t2.i, t2.j, t1.k] - self.C[t1.i, t1.j, t1.k] - self.C[t2.i, t2.j, t2.k]

        round = 1
        while True:
            if self._verbosity:
                print("-"*20)
                indices_i = np.array([t.i for t in tensors])
                indices_j = np.array([t.j for t in tensors])
                indices_k = np.array([t.k for t in tensors])
                val = np.sum(self.C[indices_i, indices_j, indices_k])
                fraction = (dual_value - val)/val
                print(f"Iteration: {round}")
                print(f"Primal Value: {val:.2f}, Dual value: {dual_value:.3f}, Duality %: {100*fraction:.2f}%")
                
            max_I_value = np.max(I)

            max_I_indices = np.unravel_index(np.argmax(I), I.shape)
            max_J_value = np.max(J)

            max_J_indices = np.unravel_index(np.argmax(J), J.shape)
            max_K_value = np.max(K)

            max_K_indices = np.unravel_index(np.argmax(K), K.shape)


            if max_I_value <= 0 and max_J_value <= 0 and max_K_value <= 0:
                if self._verbosity:
                    print("No further swaps found. Breaking...")
                    print("-"*20)
                    print("-"*50)
                break

            if max_I_value >= max_J_value and max_I_value >= max_K_value:
                if max_I_value > 0:
                    if self._verbosity:
                        print(f"Maximum value found on i-indices with value: {max_I_value} and indices: {max_I_indices}")
                    i1, i2 = max_I_indices
                    swap_I(i1, i2)


            elif max_J_value >= max_I_value and max_J_value >= max_K_value:
                if max_J_value > 0:
                    if self._verbosity:
                        print(f"Maximum value found on j-indices with value: {max_J_value} and indices: {max_J_indices}")
                    j1, j2 = max_J_indices
                    swap_J(j1, j2)

            else:
                if max_K_value > 0:
                    if self._verbosity:
                        print(f"Maximum value found on k-indices with value: {max_K_value} and indices: {max_K_indices}")
                    k1, k2 = max_K_indices
                    swap_K(k1, k2)

            round += 1
            
        x_out = np.zeros((N,N,N))
        indices_i = np.array([t.i for t in tensors])
        indices_j = np.array([t.j for t in tensors])
        indices_k = np.array([t.k for t in tensors])
        x_out[indices_i, indices_j, indices_k] = 1

        return x_out

    # def local_process(self, x, dual_value):
    #     if not self._op_solver:
    #         return x
    #     print("-"*50)
    #     print("Starting 2-OPT:")
    #     I = np.zeros((self.N, self.N))
    #     J = np.zeros((self.N, self.N))
    #     K = np.zeros((self.N, self.N))

    #     round = 1

    #     while True:
    #         print("-"*20)
    #         val = self.value(x)
    #         fraction = (dual_value - val)/val
    #         if self._verbosity:
    #             print(f"Iteration: {round}")
    #             print(f"Primal Value: {val:.2f}, Dual value: {dual_value:.3f}, Duality %: {100*fraction:.2f}%")
    #         # print("Creating Swap Cost Matrices")
    #         for i1 in range(self.N):
    #             for i2 in range(self.N):
    #                 j1, k1 = np.where(x[i1,:,:])
    #                 j2, k2 = np.where(x[i2,:,:])
    #                 I[i1, i2] = self.C[i2, j1, k1].sum() + self.C[i1, j2, k2].sum() - self.C[i1, j1, k1].sum() - self.C[i2, j2, k2].sum()


    #         for j1 in range(self.N):
    #             for j2 in range(self.N):
    #                 i1, k1 = np.where(x[:,j1,:])
    #                 i2, k2 = np.where(x[:,j2,:])
    #                 J[j1, j2] = self.C[i1, j2, k1].sum() + self.C[i2, j1, k2].sum() - self.C[i1, j1, k1].sum() - self.C[i2, j2, k2].sum()

    #         for k1 in range(self.N):
    #             for k2 in range(self.N):
    #                 i1, j1 = np.where(x[:,:,k1])
    #                 i2, j2 = np.where(x[:,:,k2])
    #                 K[k1, k2] = self.C[i1, j1, k2].sum() + self.C[i2, j2, k1].sum() - self.C[i1, j1, k1].sum() - self.C[i2, j2, k2].sum()

 
    #         max_I_value = np.max(I)

    #         max_I_indices = np.unravel_index(np.argmax(I), I.shape)
    #         max_J_value = np.max(J)

    #         max_J_indices = np.unravel_index(np.argmax(J), J.shape)
    #         max_K_value = np.max(K)

    #         max_K_indices = np.unravel_index(np.argmax(K), K.shape)
            


    #         if max_I_value <= 0 and max_J_value <= 0 and max_K_value <= 0:
    #             if self._verbosity:
    #                 print("No further swaps found. Breaking...")
    #                 print("-"*20)
    #                 print("-"*50)
    #             break

    #         if max_I_value >= max_J_value and max_I_value >= max_K_value:
    #             if max_I_value > 0:
    #                 print(f"Maximum value found on i-indices with value: {max_I_value} and indices: {max_I_indices}")
    #                 i1, i2 = max_I_indices
    #                 j1, k1 = np.where(x[i1, :, :])
    #                 j2, k2 = np.where(x[i2, :, :])
 
    #                 x[i1, j1, k1] = 0
    #                 x[i2, j2, k2] = 0
    #                 x[i2, j1, k1] = 1
    #                 x[i1, j2, k2] = 1
                    

    #         elif max_J_value >= max_I_value and max_J_value >= max_K_value:
    #             if max_J_value > 0:
    #                 print(f"Maximum value found on j-indices with value: {max_J_value} and indices: {max_J_indices}")
    #                 j1, j2 = max_J_indices
    #                 i1, k1 = np.where(x[:, j1, :])
    #                 i2, k2 = np.where(x[:, j2, :])

    #                 x[i1, j1, k1] = 0
    #                 x[i2, j2, k2] = 0
    #                 x[i1, j2, k1] = 1
    #                 x[i2, j1, k2] = 1  

    #         else:
    #             if max_K_value > 0:
    #                 print(f"Maximum value found on k-indices with value: {max_K_value} and indices: {max_K_indices}")
    #                 k1, k2 = max_K_indices
    #                 i1, j1 = np.where(x[:, :, k1])
    #                 i2, j2 = np.where(x[:, :, k2])

    #                 x[i1, j1, k1] = 0
    #                 x[i2, j2, k2] = 0
    #                 x[i1, j1, k2] = 1
    #                 x[i2, j2, k1] = 1

    #         round += 1
    #     return x
         

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
        else:
            if self._verbosity:
                print("\n")
                print("-"*50)
                print("Starting optimization...")
                print(f"Using algorithm: {self._algorithm}")
                print("-"*50)
    
            initial_point = np.random.uniform(-self._search_size, self._search_size, self.N)
            if self._algorithm == "subgradient":
                return self.subgradient_algorithm(initial_point)
            else:
                return self.nesterov_accelerated_gradient(initial_point)
     