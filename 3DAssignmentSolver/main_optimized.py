import numpy as np
from scipy.optimize import linear_sum_assignment
import sys

class Solver:
    def __init__(self, beta=0.95, learning_rate=0.1, learning_rate_scale="1/k", max_iterations=100, threshold=0.01, verbosity=True):

        self._beta = beta
        self._learning_rate = learning_rate
        self._learning_rate_scale = learning_rate_scale
        self._max_iterations = max_iterations
        self._threshold = threshold

        self._verbosity = verbosity

        if self._verbosity:
            print("\n")
            print("-" * 50)
            print("Initializing Solver with parameters:")
            print("-" * 50)
            print(f"Beta: {self._beta}")
            print(f"Learning Rate: {self._learning_rate}")
            print(f"Maximum Iterations: {self._max_iterations}")
            print(f"Stopping Threshold: {self._threshold}")
            print("-" * 50)
            print("Finished initializing Solver.")
            print("-" * 50)

    def compute_learning_rate(self, k):
        if self._learning_rate_scale == "constant":
            return self._learning_rate
        elif self._learning_rate_scale == "1/k":
            return self._learning_rate / (k + 1)
        elif self._learning_rate_scale == "1/sqrt(k)":
            return self._learning_rate / np.sqrt(k + 1)
        else:
            raise ValueError("Invalid learning rate scale specified.")

    def objective_func(self, u):

        sum_C_u = self.C + u
        max_indices = np.argmax(sum_C_u, axis=-1)

        # Solve linear assignment problem in 2D (fast)
        cost_matrix = -sum_C_u[
            np.arange(self.N)[:, None], np.arange(self.N), max_indices
        ]

        i_indices, j_indices = linear_sum_assignment(cost_matrix)

        k_indices = max_indices[i_indices, j_indices]
        objective_value = -np.sum(cost_matrix[i_indices, j_indices]) - np.sum(u)

        return objective_value, i_indices, j_indices, k_indices

    def optimize(self, C):
        self.C = C
        self.N = C.shape[0]

        if self._verbosity:
            print("\n")
            print("Starting Nesterov accelerated gradient algorithm...")
            print("-" * 50)

        u = np.zeros(self.N)
        y = np.zeros(self.N)

        k = 0

        dual_value, _, _, _ = self.objective_func(u)
        if self._verbosity:
            print("Starting greedy search:")
        i_indices_greedy, j_indices_greedy, k_indices_greedy = self.greedy()
        greedy_value = np.sum(
            self.C[i_indices_greedy, j_indices_greedy, k_indices_greedy]
        )
        i_indices_greedy, j_indices_greedy, k_indices_greedy = self.local_process(
            i_indices_greedy, j_indices_greedy, k_indices_greedy, dual_value
        )
        greedy_value_refined = np.sum(
            self.C[i_indices_greedy, j_indices_greedy, k_indices_greedy]
        )

        best_value = greedy_value_refined
        i_indices_max, j_indices_max, k_indices_max = (
            i_indices_greedy,
            j_indices_greedy,
            k_indices_greedy,
        )

        dual_bounds = [dual_value]  
        primal_bounds = [best_value]  

        fraction = (dual_value - best_value) / best_value

        beta = self._beta
        if self._verbosity:
            print("\n")
            print("Starting descent...")
        while fraction > self._threshold and k < self._max_iterations:
            if self._verbosity:
                print("-" * 50)
                print(f"Iteration {k + 1}:")
                print(
                    f"Primal value: {best_value:.3f}, Dual value: {dual_value:.3f}, Current duality %: {100*fraction:.2f}%"
                )
            dual_value, i_indices, j_indices, k_indices = self.objective_func(y)

            J = np.zeros(self.N, dtype=int)
            D = np.zeros((self.N, self.N), dtype=int)

            if j_indices.size > 0:
                j_value = j_indices[0]
                k_value = k_indices[0]

                J[:] = j_value
                np.put(
                    D,
                    np.arange(self.N) * self.N + k_indices[0],
                    self.C[:, j_value, k_value],
                )

            # Solve linear assignment problem in 2D (fast)
            k_indices = linear_sum_assignment(-D)[1]

            i_indices = np.arange(self.N)

            primal_value = np.sum(self.C[i_indices, j_indices, k_indices])

            if primal_value > best_value:
                if self._verbosity:
                    print(
                        f"Best primal solution updated from {best_value} to {primal_value}."
                    )

                i_indices_max, j_indices_max, k_indices_max = (
                    i_indices,
                    j_indices,
                    k_indices,
                )
                best_value = np.sum(self.C[i_indices_max, j_indices_max, k_indices_max])

            fraction = (dual_value - best_value) / best_value

            dual_bounds.append(dual_value)
            primal_bounds.append(best_value)

            _, _, _, k_indices = self.objective_func(y)
            subgradient = np.bincount(k_indices, minlength=self.N)

            numerator = dual_value - best_value
            denominator = np.linalg.norm(subgradient) ** 2
            learning_speed = (
                self.compute_learning_rate(k) * numerator / denominator
                if denominator != 0
                else self.compute_learning_rate(k)
            )

            y = u - learning_speed * subgradient
            u = y + beta * (y - u)

            k += 1

        if self._verbosity:
            print("-" * 50)
            print(
                f"Nesterov algorithm finished with primal value {best_value:.3f}, dual value {dual_value:.3f} and duality % {100*fraction:.2f}%"
            )
            print("-" * 50)

            print("\n")
            print("Performing local search on primal...")

        dual_value = min(dual_bounds)

        i_indices_max, j_indices_max, k_indices_max = self.local_process(
            i_indices_max, j_indices_max, k_indices_max, dual_value
        )
        best_value = np.sum(self.C[i_indices_max, j_indices_max, k_indices_max])

        dual_bounds.append(dual_value)
        primal_bounds.append(best_value)
        best_sol_coords = [i_indices_max, j_indices_max, k_indices_max]
        if self._verbosity:
            print("-" * 50)
            print(f"FINISHED OPTIMIZATION")
            print("-" * 50)
            print("SEARCH VALUES")
            print(f"Greedy value: {greedy_value}")
            print(f"Greedy + 2OPT value: {greedy_value_refined}")
            print(f"Best value: {best_value}")
            print("-" * 50)

            print("OPTIMIZATION RESULTS")
            print(
                f"Primal: {primal_bounds[-1]:.2f} <= Value: {best_value:.2f} <= Dual: {dual_bounds[-1]:.2f}, duality %: {100*(dual_bounds[-1] - primal_bounds[-1])/primal_bounds[-1]:.6f}%, feasiblity: {self.is_feasible(best_sol_coords)}"
            )
            print("-" * 50)

        return dual_bounds, primal_bounds, best_sol_coords, best_value

    def greedy(self):
        i_indices = []
        j_indices = []
        k_indices = []

        C = self.C.copy()
        for _ in range(self.N):

            max_index = np.argmax(C)
            max_i, max_j, max_k = np.unravel_index(max_index, C.shape)

            i_indices.append(max_i)
            j_indices.append(max_j)
            k_indices.append(max_k)
            C[max_i, :, :] = -sys.maxsize
            C[:, max_j, :] = -sys.maxsize
            C[:, :, max_k] = -sys.maxsize

        return i_indices, j_indices, k_indices

    def local_process(self, i_indices, j_indices, k_indices, dual_value):
        
        def swap_I(i1, i2):

            # Find the indices of i1 and i2 in the indices array
            index_a = np.where(indices[:, 0] == i1)[0][0]
            index_b = np.where(indices[:, 0] == i2)[0][0]

            # Swap the values of i1 and i2 in the indices array
            indices[index_a, 0], indices[index_b, 0] = i2, i1

            # Calculate a and b
            a = (
                self.C[i1, indices[index_a, 1], indices[index_a, 2]]
                + self.C[i2, indices[index_b, 1], indices[index_b, 2]]
            )
            b = (
                self.C[i2, indices[index_a, 1], indices[index_a, 2]]
                + self.C[i1, indices[index_b, 1], indices[index_b, 2]]
            )

            # Calculate j1, j2, k1, k2 based on the modified indices array
            j1 = indices[index_a, 1]
            j2 = indices[index_b, 1]
            k1 = indices[index_a, 2]
            k2 = indices[index_b, 2]

            # Update the I matrix
            I[i1, i2] = a - b
            I[i2, i1] = a - b

            # update J
            for idx in range(self.N):
                # update J
                # look at (idx, j1) for i2
                s = indices[indices[:, 1] == j1][0]
                t = indices[indices[:, 1] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], t[1], s[2]] + self.C[t[0], s[1], t[2]]

                J[idx, j1] = b - a
                J[j1, idx] = b - a

                # look at (idx, j2) for i1
                s = indices[indices[:, 1] == j2][0]
                t = indices[indices[:, 1] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], t[1], s[2]] + self.C[t[0], s[1], t[2]]

                J[idx, j2] = b - a
                J[j2, idx] = b - a

                # update K
                # look at (idx, k1) for i2
                s = indices[indices[:, 2] == k1][0]
                t = indices[indices[:, 2] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], s[1], t[2]] + self.C[t[0], t[1], s[2]]

                K[idx, k1] = b - a
                K[k1, idx] = b - a

                # look at (idx, k2) for i1
                s = indices[indices[:, 2] == k2][0]
                t = indices[indices[:, 2] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], s[1], t[2]] + self.C[t[0], t[1], s[2]]

                K[idx, k2] = b - a
                K[k2, idx] = b - a

                # update I, but not for i1 and i2

                if i1 != idx and i2 != idx:
                    s = indices[indices[:, 0] == i1][0]
                    t = indices[indices[:, 0] == idx][0]

                    a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                    b = self.C[t[0], s[1], s[2]] + self.C[s[0], t[1], t[2]]

                    I[i1, idx] = b - a
                    I[idx, i1] = b - a

                    s = indices[indices[:, 0] == i2][0]
                    t = indices[indices[:, 0] == idx][0]

                    a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                    b = self.C[t[0], s[1], s[2]] + self.C[s[0], t[1], t[2]]

                    I[i2, idx] = b - a
                    I[idx, i2] = b - a

        def swap_J(j1, j2):

            index_a = np.where(indices[:, 1] == j1)[0][0]
            index_b = np.where(indices[:, 1] == j2)[0][0]

            indices[index_a, 1], indices[index_b, 1] = j2, j1

            a = (
                self.C[indices[index_a, 0], j1, indices[index_a, 2]]
                + self.C[indices[index_b, 0], j2, indices[index_b, 2]]
            )
            b = (
                self.C[indices[index_a, 0], j2, indices[index_a, 2]]
                + self.C[indices[index_b, 0], j1, indices[index_b, 2]]
            )

            i1 = indices[index_a, 0]
            i2 = indices[index_b, 0]
            k1 = indices[index_a, 2]
            k2 = indices[index_b, 2]

            J[j1, j2] = a - b
            J[j2, j1] = a - b

            # update I
            for idx in range(self.N):
                # update I
                # look at (i1, idx) for j2
                s = indices[indices[:, 0] == i1][0]
                t = indices[indices[:, 0] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[t[0], s[1], s[2]] + self.C[s[0], t[1], t[2]]

                I[i1, idx] = b - a
                I[idx, i1] = b - a

                # look at (i2, idx) for j1
                s = indices[indices[:, 0] == i2][0]
                t = indices[indices[:, 0] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[t[0], s[1], s[2]] + self.C[s[0], t[1], t[2]]

                I[i2, idx] = b - a
                I[idx, i2] = b - a

                # update K
                # look at (idx, k1) for j2
                s = indices[indices[:, 2] == k1][0]
                t = indices[indices[:, 2] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], s[1], t[2]] + self.C[t[0], t[1], s[2]]

                K[idx, k1] = b - a
                K[k1, idx] = b - a

                # look at (idx, k2) for j1
                s = indices[indices[:, 2] == k2][0]
                t = indices[indices[:, 2] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], s[1], t[2]] + self.C[t[0], t[1], s[2]]

                K[idx, k2] = b - a
                K[k2, idx] = b - a

                # update J, but not for j1 and j2
                if j1 != idx and j2 != idx:
                    s = indices[indices[:, 1] == j1][0]
                    t = indices[indices[:, 1] == idx][0]

                    a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                    b = self.C[s[0], t[1], s[2]] + self.C[t[0], s[1], t[2]]

                    J[j1, idx] = b - a
                    J[idx, j1] = b - a

                    s = indices[indices[:, 1] == j2][0]
                    t = indices[indices[:, 1] == idx][0]

                    a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                    b = self.C[s[0], t[1], s[2]] + self.C[t[0], s[1], t[2]]

                    J[j2, idx] = b - a
                    J[idx, j2] = b - a

        def swap_K(k1, k2):
            index_a = np.where(indices[:, 2] == k1)[0][0]
            index_b = np.where(indices[:, 2] == k2)[0][0]

            indices[index_a, 2], indices[index_b, 2] = k2, k1

            a = (
                self.C[indices[index_a, 0], indices[index_a, 1], k1]
                + self.C[indices[index_b, 0], indices[index_b, 1], k2]
            )
            b = (
                self.C[indices[index_a, 0], indices[index_a, 1], k2]
                + self.C[indices[index_b, 0], indices[index_b, 1], k1]
            )

            K[k1, k2] = a - b
            K[k2, k1] = a - b

            i1 = indices[index_a, 0]
            i2 = indices[index_b, 0]
            j1 = indices[index_a, 1]
            j2 = indices[index_b, 1]

            # update I
            for idx in range(self.N):
                # update I
                # look at (i1, idx) for k2
                s = indices[indices[:, 0] == i1][0]
                t = indices[indices[:, 0] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[t[0], s[1], s[2]] + self.C[s[0], t[1], t[2]]

                I[i1, idx] = b - a
                I[idx, i1] = b - a

                # look at (i2, idx) for k1
                s = indices[indices[:, 0] == i2][0]
                t = indices[indices[:, 0] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[t[0], s[1], s[2]] + self.C[s[0], t[1], t[2]]

                I[i2, idx] = b - a
                I[idx, i2] = b - a

                ###

                # update J
                # look at (j1, idx) for k2
                s = indices[indices[:, 1] == j1][0]
                t = indices[indices[:, 1] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], t[1], s[2]] + self.C[t[0], s[1], t[2]]

                J[j1, idx] = b - a
                J[idx, j1] = b - a

                # look at (j2, idx) for k1
                s = indices[indices[:, 1] == j2][0]
                t = indices[indices[:, 1] == idx][0]

                a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                b = self.C[s[0], t[1], s[2]] + self.C[t[0], s[1], t[2]]

                J[j2, idx] = b - a
                J[idx, j2] = b - a

                # update K
                #
                if k1 != idx and k2 != idx:
                    s = indices[indices[:, 2] == k1][0]
                    t = indices[indices[:, 2] == idx][0]

                    a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                    b = self.C[s[0], s[1], t[2]] + self.C[t[0], t[1], s[2]]
                    K[k1, idx] = b - a
                    K[idx, k1] = b - a

                    s = indices[indices[:, 2] == k2][0]
                    t = indices[indices[:, 2] == idx][0]

                    a = self.C[s[0], s[1], s[2]] + self.C[t[0], t[1], t[2]]
                    b = self.C[s[0], s[1], t[2]] + self.C[t[0], t[1], s[2]]

                    K[k2, idx] = b - a
                    K[idx, k2] = b - a

        indices = np.column_stack((i_indices, j_indices, k_indices))
        
        I = np.zeros((self.N, self.N))
        J = np.zeros((self.N, self.N))
        K = np.zeros((self.N, self.N))

        for idx1 in range(self.N):
            for idx2 in range(self.N):
                t1 = indices[indices[:, 0] == idx1][0]
                t2 = indices[indices[:, 0] == idx2][0]
                I[t1[0], t2[0]] = (
                    self.C[t2[0], t1[1], t1[2]]
                    + self.C[t1[0], t2[1], t2[2]]
                    - self.C[t1[0], t1[1], t1[2]]
                    - self.C[t2[0], t2[1], t2[2]]
                )
                J[t1[1], t2[1]] = (
                    self.C[t1[0], t2[1], t1[2]]
                    + self.C[t2[0], t1[1], t2[2]]
                    - self.C[t1[0], t1[1], t1[2]]
                    - self.C[t2[0], t2[1], t2[2]]
                )
                K[t1[2], t2[2]] = (
                    self.C[t1[0], t1[1], t2[2]]
                    + self.C[t2[0], t2[1], t1[2]]
                    - self.C[t1[0], t1[1], t1[2]]
                    - self.C[t2[0], t2[1], t2[2]]
                )
        
        round = 1
        while True:
            
            if self._verbosity:
                print("-" * 50)
                val = np.sum(self.C[indices[:, 0], indices[:, 1], indices[:, 2]])
                fraction = (dual_value - val) / val
                print(f"Iteration: {round}")
                print(
                    f"Primal Value: {val:.2f}, Dual value: {dual_value:.3f}, Duality %: {100*fraction:.2f}%"
                )

            max_I_value = np.max(I)

            max_I_indices = np.unravel_index(np.argmax(I), I.shape)
            max_J_value = np.max(J)

            max_J_indices = np.unravel_index(np.argmax(J), J.shape)
            max_K_value = np.max(K)

            max_K_indices = np.unravel_index(np.argmax(K), K.shape)

            max_values = [I[max_I_indices], J[max_J_indices], K[max_K_indices]]
            max_indices = [max_I_indices, max_J_indices, max_K_indices]

            max_index = np.argmax(max_values)
            max_value = max_values[max_index]
            max_indices = max_indices[max_index]

            
            if max_value > 0:
        
                if max_index == 0:
                    if self._verbosity:
                        print(
                            f"Maximum value found on i-indices with value: {max_I_value} and indices: {max_I_indices}"
                        )
                    i1, i2 = max_indices
                    swap_I(i1, i2)
                elif max_index == 1:
                    if self._verbosity:
                        print(
                            f"Maximum value found on j-indices with value: {max_J_value} and indices: {max_J_indices}"
                        )
                    j1, j2 = max_indices
                    swap_J(j1, j2)
                else:
                    if self._verbosity:
                        print(
                            f"Maximum value found on k-indices with value: {max_K_value} and indices: {max_K_indices}"
                        )
                    k1, k2 = max_indices
                    swap_K(k1, k2)

            else:
                if self._verbosity:
                    print("No further swaps found. Breaking...")
                    print("-" * 50)
                    print("\n")
                break

            round += 1

        return indices[:, 0], indices[:, 1], indices[:, 2]

    def is_feasible(self, x):
        if (
            len(np.unique(x[0])) + len(np.unique(x[1])) + len(np.unique(x[2]))
            < 3 * self.N
        ):
            return False
        return True
