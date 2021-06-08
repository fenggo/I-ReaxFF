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
也可以使用命令安装openmpi

``` sudo apt install openmpi-bin ```

## 安装 lapack
下载：http://www.netlib.org/lapack
Cd BLAS/Src
make
cp make.example.inc make.inc
并修改以下几行：
BLASLIB      =  ../../librefblas.a   # 将要创建一个librefblas.a
LAPACKLIB    =  liblapack.a
TMGLIB       =  libtmglib.a
LAPACKELIB   =  liblapacke.a

Make

## 安装 BLACS
http://www.netlib.org/blacs下载MPIBLACS
在BMAKES文件夹中拷出Bmake.MPI-LINUX Bmake.inc

参见：How do I build BLACS with Open MPI http://www.open-mpi.org/faq/?category=mpi-apps

## 修改Bmake.inc
```
# Section 1:
BLACS文件夹路径
BTOPdir = /home/feng/siesta/mathlib/BLACS
# Section 2:
# Ensure to use MPI for the communication layer
   COMMLIB = MPI
# The MPIINCdir macro is used to link in mpif.h and
# must contain the location of Open MPI's mpif.h.  
# The MPILIBdir and MPILIB macros are irrelevant 
# and should be left empty.
   MPIdir = /path/to/openmpi-1.6.5
   MPILIBdir =
   MPIINCdir = $(MPIdir)/include
   MPILIB =

# Section 3:
# Set these values:
   SYSINC =
   INTFACE = -Df77IsF2C
   SENDIS =
   BUFF =
   TRANSCOMM = -DUseMpi2
   WHATMPI =
   SYSERRORS =
# Section 4:
# You may need to specify the full path to
# mpif77 / mpicc if they aren't already in
# your path.
   F77            = mpif77
   F77LOADFLAGS   = 
   CC             = mpicc
   CCLOADFLAGS    = 
```
Make mpi

## 安装 scalapack
修改SLmake.inc
参见：http://www.open-mpi.org/faq/?category=mpi-apps

例如：
```
BLACSdir      = /path to/BLACS/LIB
BLASLIB       = -L/path to/lapack-3.4.2 -lrefblas
LAPACKLIB     = -L/path to/lapack-3.4.2 -llapack
LIBS          = $(LAPACKLIB) $(BLASLIB)

USEMPI        = -DUsingMpiBlacs
SMPLIB        = 
BLACSFINIT    = $(BLACSdir)/blacsF77init_MPI-LINUX-0.a
BLACSCINIT    = $(BLACSdir)/blacsCinit_MPI-LINUX-0.a
BLACSLIB      = $(BLACSdir)/blacs_MPI-LINUX-0.a
#TESTINGdir    = $(home)/TESTING
```
Then type : make

## 编译siesta
将Src/MPI中全部拷到 Obj/MPI中，并执行

```make```

在Obj文件夹中执行：
sh ../Src/obj_setup.sh

## 修改 arch.make

添加或修改LIB = 下面的内容
FC=mpif90
FPPFLAGS= -DFC_HAVE_FLUSH -DFC_HAVE_ABORT -DMPI

BLAS_LIBS=       -L/path/lapack-3.4.2/ -lrefblas
LAPACK_LIBS=     -L/path/lapack-3.4.2/ -llapack -ltmglib
BLACS_LIBS=     -L/path/BLACS/LIB/blacs_MPI-LINUX-0.a 
SCALAPACK_LIBS= -L/path/scalapack-2.0.2/ -lscalapack 

COMP_LIBS= dc_lapack.a linalg.a

MPI_INTERFACE=libmpi_f90.a
MPI_INCLUDE=/home/feng/scisoft/openmpi-gnu/include
测试：
``` mpirun -np <nproc> siesta < input.fdf > output ```
