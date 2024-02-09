import gurobipy as gp
from gurobipy import GRB
import os
import sys
import json

class GurobiSolver:
    """
    Gurobi solver for the 3D Assignment Problem.

    Parameters:
        address: str
            Path to the JSON file containing Gurobi parameters.

    Attributes:
        env: Gurobi environment
            Gurobi environment initialized with parameters loaded from the JSON file.
        model: Gurobi model
            Gurobi optimization model.

    Methods:
        optimize(C):
            Solve the 3D Assignment Problem for the given cost matrix.

    """
    def __init__(self, address):
        """
        Initialize the Gurobi solver.

        Parameters:
            address: str
                Path to the JSON file containing Gurobi parameters.
        """
        # Save the original stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Open a null device for stdout and stderr
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull

            # Load parameters from JSON file and initialize Gurobi environment and model
            with open(address) as f:
                params = json.load(f)

            self.env = gp.Env(params=params)
            self.model = gp.Model(env=self.env)

        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    def optimize(self, C):
        """
        Optimize the 3D Assignment Problem for the given cost matrix.

        Parameters:
            C: numpy array
                Cost matrix.

        Returns:
            obj_val: float
                Objective value of the optimized solution.
            selected_indices: list of tuples
                Indices of the selected elements in the solution matrix.
        """
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
