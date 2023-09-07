from setuptools import find_packages, setup
setup(
    name='normalizing_flows',
    packages=find_packages(include=['normalizing_flows']),
    version='0.1.0',
    description='An implementation of Normalizing Flows using Tensorflow Probability',
    author='Arshak',
    license='QOO',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)