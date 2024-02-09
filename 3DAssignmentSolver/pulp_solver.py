from pulp import LpProblem, lpSum, LpVariable, LpMaximize, PULP_CBC_CMD

class PulpSolver:
    """
    PulpSolver class for solving the 3D assignment problem using the PuLP library.

    Attributes:
        None

    Methods:
        optimize(C): Solves the 3D assignment problem.
    """
    def __init__(self):
        """
        Initializes the PulpSolver class.
        """
        pass

    def optimize(self, C):
        """
        Solves the 3D assignment problem using the PuLP library.

        Parameters:
            C (numpy array): Cost matrix.

        Returns:
            tuple: A tuple containing the objective value and None.
        """
        N = C.shape[0]

        problem = LpProblem("3D_Assignment_Problem", LpMaximize)
        x = [
            [
                [LpVariable(f"x_{i}_{j}_{k}", cat="Binary") for k in range(N)]
                for j in range(N)
            ]
            for i in range(N)
        ]

        problem += lpSum(
            C[i][j][k] * x[i][j][k]
            for i in range(N)
            for j in range(N)
            for k in range(N)
        )

        for i in range(N):
            problem += lpSum(x[i][j][k] for j in range(N) for k in range(N)) == 1

        for j in range(N):
            problem += lpSum(x[i][j][k] for i in range(N) for k in range(N)) == 1

        for k in range(N):
            problem += lpSum(x[i][j][k] for i in range(N) for j in range(N)) == 1

        problem.solve(PULP_CBC_CMD(msg=False))

        val = lpSum(
            C[i, j, k] * x[i][j][k].value()
            for i in range(N)
            for j in range(N)
            for k in range(N)
        ).value()

        return val, None  # just for ease of utils
