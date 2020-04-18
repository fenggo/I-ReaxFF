#!/usr/bin/env python
from irff.mdtodata import MDtoData
from os import getcwd


cwd = getcwd()
d = MDtoData(structure='siesta',dft='siesta',dic=cwd,batch=10000)
d.get_traj()
d.close()

