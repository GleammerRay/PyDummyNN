from setuptools import find_packages, setup

setup(
    name='dummynn',    
    packages=find_packages(include=['dummynn', 'dummynn.*']),
    version='0.5.0',
    description='A silly neural network algorithm for Python',
    author='Gleammer',
    license='MIT',
)
