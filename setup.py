from setuptools import setup, find_packages
# from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize


'''
install with commond 
  "python setup.py build_ext --inplace"
  "python setup install --user"
or with command
*  version 1.7.1 
  Remove of dependence of pandas.

*  version 1.7.0 
  Force learning with TensorFlow.

*  version 1.6.2 
  Bus fixes.

*  version 1.6.1 
  Add force learning and PyTorch backend.

*  version 1.6 
  Add a penalty term for translation-invariant.
'''


__version__ = '1.7.1'
install_requires = ['numpy','ase','tensorflow','matplotlib','paramiko','argh','scikit-learn','cython']
url = "https://gitee.com/fenggo/I-ReaxFF"


setup(name="irff",
      version=__version__,
      description="Intelligent Reactive Force Field",
      author="FengGo",
      author_email='fengguo@lcu.edu.cn',
      url=url,
      download_url='{}/archive/{}.tar.gz'.format(url, __version__),
      license="LGPL",
      packages= find_packages(),
      package_data={'': ['*.gen','*.pyx']},
      ext_modules=cythonize(['irff/neighbor.pyx','irff/getNeighbor.pyx'],annotate=True))


