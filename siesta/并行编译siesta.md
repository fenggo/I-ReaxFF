# 使用gfortran和openmpi并行编译SIESTA
## Openmpi安装
1. ./configure --prefix=/home/feng/siesta/mathlib/openmpi-gnu CC=gcc CXX=g++ F77=gfortran FC=gfortran
 对于intel ./configure --prefix=/usr/local/openmpi-1.4.3 CC=icc CXX=icpc F77=ifort FC=ifort
2.  make all 
3.  make install
4. 打开 ～/.bashrc 添加环境变量
   
   export PATH=/usr/local/openmpi-1.4.3/bin:$PATH
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/openmpi-1.4.3/lib

5. source ~/.bashrc（重新打开终） 并验证    which mpicc
                            which  mpic＋＋
                            which mpif77
                            which mpif90
安装 lapack
下载：http://www.netlib.org/lapack
Cd BLAS/Src
make
cp make.example.inc make.inc
并修改以下几行：
BLASLIB      =  /your/path/to/lapack-3.4.2/librefblas.a   # 将要创建一个librefblas.a
LAPACKLIB    =  liblapack.a
TMGLIB       =  libtmglib.a
LAPACKELIB   =  liblapacke.a

Make
安装 BLACS
http://www.netlib.org/blacs下载MPIBLACS
在BMAKES文件夹中拷出Bmake.MPI-LINUX Bmake.inc

参见：How do I build BLACS with Open MPI http://www.open-mpi.org/faq/?category=mpi-apps
修改Bmake.inc
# Section 1:
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

# Section 2:
# Set these values:
   SYSINC =
   INTFACE = -Df77IsF2C
   SENDIS =
   BUFF =
   TRANSCOMM = -DUseMpi2
   WHATMPI =
   SYSERRORS =
# Section 3:
# You may need to specify the full path to
# mpif77 / mpicc if they aren't already in
# your path.
   F77            = mpif77
   F77LOADFLAGS   = 
   CC             = mpicc
   CCLOADFLAGS    = 
Make mpi
安装 scalapack
修改SLmake.inc
参见：http://www.open-mpi.org/faq/?category=mpi-apps

# Make sure you follow the instructions to build BLACS with Open MPI,
# and put its location in the following.

   BLACSdir      = <path where you installed BLACS>

# The MPI section is commented out.  Uncomment it. The wrapper
# compiler will handle SMPLIB, so make it blank. The rest are correct
# as is.

   USEMPI        = -DUsingMpiBlacs
   SMPLIB        = 
   BLACSFINIT    = $(BLACSdir)/blacsF77init_MPI-$(PLAT)-$(BLACSDBGLVL).a
   BLACSCINIT    = $(BLACSdir)/blacsCinit_MPI-$(PLAT)-$(BLACSDBGLVL).a
   BLACSLIB      = $(BLACSdir)/blacs_MPI-$(PLAT)-$(BLACSDBGLVL).a
   TESTINGdir    = $(home)/TESTING

# The PVMBLACS setup needs to be commented out.

   #USEMPI        =
   #SMPLIB        = $(PVM_ROOT)/lib/$(PLAT)/libpvm3.a -lnsl -lsocket
   #BLACSFINIT    =
   #BLACSCINIT    =
   #BLACSLIB      = $(BLACSdir)/blacs_PVM-$(PLAT)-$(BLACSDBGLVL).a
   #TESTINGdir    = $(HOME)/pvm3/bin/$(PLAT)

# Make sure that the BLASLIB points to the right place.  We built this
# example on Solaris, hence the name below.  The Linux version of the
# library (as of this writing) is blas_LINUX.a.

   BLASLIB       = $(LAPACKdir)/blas_solaris.a

# You may need to specify the full path to mpif77 / mpicc if they
# aren't already in your path.
我改的：
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
Then type : make
NetCDF编译 
cd netcdf-4.1.3
 ./configure --disable-dap --disable-netcdf-4 --prefix=/home/feng/siesta/mathlib/netcdf

编译siesta
将Src/MPI中全部拷到 Obj/MPI中，make
在Obj文件夹中执行：
sh ../Src/obj_setup.sh
../Src/configure --enable-mpi(详见--help)
修改 arch.make, 我的：
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
mpirun -np <nproc> siesta < input.fdf > output