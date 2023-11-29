一、编译LAMMPS可执行文件
1. 安装依赖的库文件，打开终端，在终端中输入命令：
```shell
sudo apt install libfftw3-3 libfftw3-dev libjpeg-dev libpng-dev
```
如果软件名字对不上，可以使用"sudo apt search libfftw3"命令，搜索相近库，进行安装。

2. 将lammps/src/MAKE/MACHINES文件夹中的“Makefile.ubuntu”拷贝到lammps/src/MAKE/目录下。

3. 在lammps/src目录中运行命令
```shell
make yes-reaxff
make ubuntu
```
二、编译LAMMPS Python接口
1. 在lammps/src目录中运行命令
```shell
make ubuntu mode=shlib
make install-python
```
在python环境下运行
```python
from lammps import lammps
```
如能加载没有报错信息,安装成功。