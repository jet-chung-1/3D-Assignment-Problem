from setuptools import setup, find_packages

setup(
    name='AssignmentSolver',
    version='1.0',
    py_modules=['main', 'utils', 'pulp_solver', 'gurobi_solver'],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'gurobipy',
        'pulp'
    ],
)
