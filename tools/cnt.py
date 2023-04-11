#!/usr/bin/env python
from ase.build import nanotube
# cnt1 = nanotube(6, 0, length=4)

cnt = nanotube(3, 3, length=3, bond=1.4, symbol='C')
cnt.write('cnt333.gen')


