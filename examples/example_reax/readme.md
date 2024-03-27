### Example of Train ReaxFF

1. ml_train_reax.py: python script use machine learning algorithm for training ReaxFF.
```shell
./ml_train_reax.py --s=10000
```
2. train_reax.py: python script use gradient based algorithm for training ReaxFF.
   usage:
```shell
./train_reax.py --s=10000
```
3. json_to_ffiled.py: python script convert ffield.json file to ffield and control file, which is used by lammps.
   usage:
```shell
./json_to_ffield.py
```
4. lmd.py: python script call lammps to run MD.
   usage:
```shell
./lmd.py nvt --s=3000 --g=tkx2.gen --m=reaxff --n=4
```