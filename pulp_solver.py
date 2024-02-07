from pulp import LpProblem, lpSum, LpVariable, LpMaximize, PULP_CBC_CMD


class PulpSolver:
    def __init__(self):
        pass

    def optimize(self, C):

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
        return val
