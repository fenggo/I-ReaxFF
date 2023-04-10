To install GULP on most Unix/Linux machines :

(1) Go to Src/

(2) Type "./mkgulp -h"     该命令会列出所有可用选项
    
NB: There are several flags that can be added depending on what options are required. The main options are:
*        -h => print help text to list the options below
*        -m => parallel compilation with MPI
*        -d => compile with debug options
*        -f => compile with FFTW3 for Smooth Particle Mesh Ewald
*        -k => compile with OpenKIM
*        -p => compile with Plumed plug-in
*        -c ARG => change the compiler to "ARG". gfortran (default) and intel are supported.
*        -j ARG => specify the number of cores to be used for task (i.e. make -j)
*        -t ARG => change the task to ARG. gulp (default), clean, lib, tar, fox-clean

(3)找到下面部分内容， 修改“mkgulp”脚本内容：

```bash
case $compiler in
  gfortran)
...
# 215行，else后面的USER部分
#--USER--Start
               echo 'SLIBS= -L/home/feng/mathlib/scalapack -lscalapack' >> makefile
               echo 'MLIBS= -L/home/feng/mathlib/lapack  -llapack -ltmglib -lrefblas' >> makefile
#--USER--End
```

（上面的库在“并行编译siesta.md”里面已经有编译方法，如果siesta编译好了，找到库文件所在路径就行了。OpenMPI编译参见“并行编译siesta.md”。）

(4) 运行“mkgulp”脚本，加“-m”选项进行并行编译：

```bash
./mkgulp -m
```

(5) 测试GULP

```bash
nohup mpirun -n 8 gulp<inp-gulp>gulp.out 2>&1 & 
```