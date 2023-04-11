#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.AtomDance import AtomDance
from irff.zmatrix import zmat_to_atoms
# get_ipython().run_line_magic('matplotlib', 'inline')


zmat = [
[ 'O',  6,  -1,  -1,  -1,   0.0000,  0.0000,   0.0000 ],
[ 'N',  4,   6,  -1,  -1,   1.4901,  0.0000,   0.0000 ],
[ 'N',  5,   4,   6,  -1,   1.3554,120.1679,   0.0000 ],
[ 'O',  7,   4,   6,   5,   1.2807,118.8074,   2.7061 ],
[ 'C',  0,   5,   4,   6,   1.4933,128.1317, 163.3464 ],
[ 'H',  2,   0,   5,   4,   1.0941,117.1877, -45.5241 ],
[ 'H',  3,   0,   5,   4,   1.2500,116.6197,-167.4330 ],
[ 'H',  8,   0,   5,   4,   1.0964,118.7857,  80.5217 ],
[ 'C',  1,   5,   4,   6,   1.5971,123.6618,  -4.8899 ],
[ 'H',  9,   1,   5,   4,   1.0874,115.2583,  50.7324 ],
[ 'N', 10,   1,   5,   4,   1.4651,137.2663,-179.6538 ],
[ 'H', 11,  10,   1,   5,   1.0583,133.5558, 167.7767 ],
[ 'H', 12,  10,   1,   5,   1.0660,117.1625,  15.0633 ],
[ 'H', 13,   1,   5,   4,   1.1038,107.3054, -50.3790 ],
 ]

atoms = zmat_to_atoms(zmat,resort=False)

atoms.write('hmx2-0.gen')
# view(atoms)


