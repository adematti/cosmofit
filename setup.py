import os
import sys
from setuptools import setup


package_basename = 'cosmofit'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='adematti',
      author_email='',
      description='Package for cosmological constraints',
      license='BSD3',
      url='http://github.com/adematti/cosmofit',
      install_requires=['numpy', 'scipy'],
      extras_require={'extras': ['cosmoprimo']},
      packages=[package_basename])
