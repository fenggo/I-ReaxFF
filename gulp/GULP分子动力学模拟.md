## GULP分子动力学模拟
### 1、使用“gmd.py”脚本制作GULP输入文件：
```bash
./gmd.py nvt --s=1 --g=xxx.gen
```
脚本执行完毕可生成inp-gulp输入文件，其中“xxx.gen”为结构文件。根据需要编辑“inp-gulp”文件，使用
```bash
nohup mpirun -n 8 gulp<inp-gulp>gulp.out 2>&1 & 
```
并行运行命令进行分子动力学模拟。

