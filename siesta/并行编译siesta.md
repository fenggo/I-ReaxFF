# 使用gfortran和openmpi并行编译SIESTA

## SIESTA下载
https://gitlab.com/siesta-project/siesta

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

或用命令安装
``` sudo apt install openmpi ```

## 安装 lapack
```sudo apt install libblas```
```sudo apt install libmkl-scalapack-lp64```

## 安装 BLACS
```sudo apt install libmkl-blacs-openmpi-lp64```


## 安装 scalapack
``` sudo apt install libmkl-scalapack-lp64 ```


## 编译siesta
将Src/MPI中全部拷到 Obj/MPI中，并执行

```make```

###  修改 arch.make

添加或修改LIB = 下面的内容
FC=mpif90
FPPFLAGS= -DFC_HAVE_FLUSH -DFC_HAVE_ABORT -DMPI

BLAS_LIBS=         
LAPACK_LIBS=     
BLACS_LIBS=  -lmkl-blacs-openmpi-lp64    
SCALAPACK_LIBS=  -lmkl-scalapack-lp64 

COMP_LIBS= dc_lapack.a linalg.a

MPI_INTERFACE=libmpi_f90.a
MPI_INCLUDE=/home/feng/scisoft/openmpi-gnu/include
测试：
mpirun -np <nproc> siesta < input.fdf > output