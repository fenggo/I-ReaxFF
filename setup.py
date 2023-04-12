from setuptools import setup, find_packages
# from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize


'''
install with commond 
  "python setup.py build_ext --inplace"
  "python setup install --user"
'''


__version__ = '1.5.2'
install_requires = ['numpy','ase','tensorflow','matplotlib','paramiko']
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
      package_data={'': ['*.gen']},
      ext_modules=cythonize(['irff/neighbor.pyx','irff/getNeighbor.pyx'],annotate=True))


