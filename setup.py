import os
import sys
from setuptools import setup


package_basename = 'cosmofit'
package_dir = os.path.join(os.path.dirname(__file__), package_basename)
sys.path.insert(0, package_dir)
import _version
version = _version.__version__


def get_yaml_files():
    for section in ['likelihoods', 'theories', 'samplers', 'profilers', 'emulators']:
        for root, dirs, files in os.walk(os.path.join(package_dir, section)):
            for file in files:
                if file.endswith('.yaml'):
                    yield os.path.relpath(os.path.join(root, file), package_dir)


setup(name=package_basename,
      version=version,
      author='adematti',
      author_email='',
      description='Package for cosmological constraints',
      license='BSD3',
      url='http://github.com/adematti/cosmofit',
      install_requires=['numpy', 'scipy', 'tabulate', 'mpytools @ git+https://github.com/adematti/mpytools'],
      extras_require={'plotting': ['getdist', 'anesthetic'], 'jax': ['jax[cpu]']},
      packages=[package_basename],
      package_data={'cosmofit': list(get_yaml_files())},
      entry_points={'console_scripts': ['cosmofit=cosmofit.__main__:main']})
