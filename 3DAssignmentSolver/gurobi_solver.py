import gurobipy as gp
from gurobipy import GRB
import json

class GurobiSolver:
    def __init__(self, address):
        with open(address) as f:
            params = json.load(f)

        self.env = gp.Env(params=params)
        self.model = gp.Model(env=self.env)

    def optimize(self, C):
        N = C.shape[0]

        model = gp.Model("3D_Assignment_Problem", env=self.env)

        x = [
            [
                [
                    model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k}")
                    for k in range(N)
                ]
                for j in range(N)
            ]
            for i in range(N)
        ]

        model.setObjective(
            gp.quicksum(
                C[i, j, k] * x[i][j][k]
                for i in range(N)
                for j in range(N)
                for k in range(N)
            ),
            GRB.MAXIMIZE,
        )

        for i in range(N):
            model.addConstr(
                gp.quicksum(x[i][j][k] for j in range(N) for k in range(N)) == 1
            )

        for j in range(N):
            model.addConstr(
                gp.quicksum(x[i][j][k] for i in range(N) for k in range(N)) == 1
            )

        for k in range(N):
            model.addConstr(
                gp.quicksum(x[i][j][k] for i in range(N) for j in range(N)) == 1
            )

        model.Params.outputFlag = 0  # Disable Gurobi output

        model.optimize()
        
        selected_indices = [(i, j, k) for i in range(N) for j in range(N) for k in range(N) if x[i][j][k].x == 1]
        return model.objVal, selected_indices