# 使用gfortran和openmpi并行编译QE

## QE下载
https://github.com/QEF/q-e/releases

## 安装必要的编译软件
(默认使用deepin linux系统)
```
sudo apt install build-essential  
sudo apt install g++
sudo apt install gfortran
```

## Openmpi安装
```
1 ./configure --prefix=/home/feng/siesta/mathlib/openmpi-gnu CC=gcc CXX=g++ F77=gfortran FC=gfortran
 对于intel ./configure --prefix=/home/feng/siesta/mathlib/openmpi-intel CC=icc CXX=icpc F77=ifort FC=ifort
2  make all 
3  make install
4 打开 ～/.bashrc 添加环境变量,用vi打开
vi ~/.bashrc
   export PATH=/home/feng/siesta/mathlib/openmpi-gnu/bin:$PATH
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feng/siesta/mathlib/openmpi-gnu/lib

5 source ~/.bashrc（重新打开终） 并验证    which mpicc
                            which  mpic＋＋
                            which mpif77
                            which mpif90
```
也可以使用命令安装openmpi, 不建议，推荐下载编译openmpi-2.0版本。

## QE安装
```
./configure
make all
```