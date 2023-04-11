## GULP分子动力学模拟
### 1、使用“gmd.py”脚本制作GULP输入文件：
```bash
./gmd.py nvt --s=1 --g=xxx.gen
```
扩展超胞

```bash
./gmd.py nvt --s=1 --g=xxx.gen --x=2 --y=2 --z=2
```
扩展后为 2 $\times$ 2 $\times$ 2 超胞

脚本执行完毕可生成inp-gulp输入文件，其中“xxx.gen”为结构文件。根据需要编辑“inp-gulp”文件，使用
```bash
nohup mpirun -n 8 gulp<inp-gulp>gulp.out 2>&1 & 
```
并行运行命令进行分子动力学模拟。

### 2、使用“gmd.py”脚本绘制温度、势能、压强随时间变化曲线：
```bash
./gmd.py plot --o=gulp.out 
```