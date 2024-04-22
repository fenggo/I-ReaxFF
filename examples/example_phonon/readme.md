# Phononpy和gulp计算声子谱和力常数（利用siesta interface）  

先将结构文件转换为siesta输入文件

```sh
./smd.py wi --g=gulp.cif
```

生成位移文件
```sh
phonopy --siesta -c=in.fdf -d --dim="6 6 1" --amplitude=0.02
```

### Using the “get_force.py” script to compute the forces through GULP and ReaxFF-nn.

```sh
./phonon_force.py
or 
./phonon_force.py --n=1 # use n to specify which supercell to be calculated
```

Alterativly, to get the siesta force, using mpirun commond instead:(如果使用GULP计算，该步骤不需要)
```sh
mpirun -n 8 siesta<supercell.fdf>siesta.out
```
### Then create the required FORCE_SETS file

```sh
phonopy -f Forces-001.FA Forces-002.FA ... Forces-00n.FA --siesta
``` 

### specify the path in the Brillouin zone
Then specify the path in the Brillouin zone you are interested in (see the phonopy documentation). Then post-process the phonopy data, providing the dimensions of the the supercell repeat either on the command line or in the settings file (a DIM file):
as an example of graphene

```sh
phonopy --siesta -c in.fdf -p --dim="6 6 1" --band="0.0 0.0 0.0 1/4 0.0 0.0  0.5 0.0 0.0  2/3 -1/3 1/2 1/3 -1/6 0.0  0.0 0.0 0.0"
```
### Finally create a band structure in gnuplot format
```sh
phonopy-bandplot --gnuplot band.yaml > band.dat
```
### 使用Phonopy计算二阶力常数
```sh
phonopy --writefc --full-fc
```


* 此时计算的二阶力常数的长度单位是Unit of length: au
需要将其力除以0.529转换成$\AA$。可使用如下Python脚本进行转换：

```python
f1 = open('FORCE_CONSTANTS_2ND','w')

lines = f0.readlines()
for line in lines:
    l = line.split()
    if len(l)==3:
       print('  {:17.12f} {:17.12f} {:17.12f}'.format(float(l[0])/0.52917721,
             float(l[1])/0.52917721,float(l[2])/0.52917721),file=f1)
    else:
       print(line,end=' ',file=f1)
  
f0.close()
f1.close()
```

保存成force_unit.py文件，然后执行脚本：
```sh
./force_unit.py
```
