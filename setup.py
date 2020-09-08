from setuptools import setup, find_namespace_packages

setup(name='spline_slam',
      version='0.1',
      description='B-spline surface SLAM',
      url='https://github.com/c2sr/spline_slam',
      author='Romulo T. Rodrigues',
      licence='GPLv3',
      packages=find_namespace_packages(include=['spline_slam.*']),
      zip_safe=False)
