from setuptools import setup, find_packages

setup(
    name='AssignmentSolver',
    version='1.0',
    py_modules=['main', 'utils'],
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jet-chung-1/3D-Assignment-Problem,
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'seaborn',
        'matplotlib',
        'timeit'
    ],
)
