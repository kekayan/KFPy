from setuptools import setup

setup(
    name='roukf',
    version='0.1.0',
    packages=['roukf'],
    package_data={
        'roukf': ['*.so'],
    },
    include_package_data=True,
    install_requires=[
        'numpy',
    ],
    author='Gonzalo Maso Talou',
    author_email='g.masotalou@auckland.ac.nz',
    description='ROUKF - Kalman filter for parameter estimation',
)
