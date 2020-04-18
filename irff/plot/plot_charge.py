#!/usr/bin/env python
from irff.mdtodata import MDtoData
from irff.qeq import qeq
from irff.reaxfflib import read_lib,write_lib
from os import getcwd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
# import handout


# doc  = handout.Handout('./')

# cwd  = getcwd()
iatom  = 0
direc  = '/home/gfeng/siesta/train/nm7_9'
d      = MDtoData(structure='siesta',dft='siesta',direc=direc,batch=100)
images = d.get_images()
q      = d.qs[:,iatom]

p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
Qe= qeq(p=p,atoms=images[0])

q_    = []
for A in images:
    # print('*  get charges of batch {0}/{1} ...\r'.format(nf,self.batch),end='\r')
    Qe.calc(A)
    qr = Qe.q[:-1]
    q_.append(qr[iatom])



p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield_best')
Qe= qeq(p=p,atoms=images[0])

qb    = []
for A in images:
    # print('*  get charges of batch {0}/{1} ...\r'.format(nf,self.batch),end='\r')
    Qe.calc(A)
    qr = Qe.q[:-1]
    qb.append(qr[iatom])

# print(d.qs.shape)



plt.figure()
plt.ylabel('Charges Distribution of %s' %d.atom_name[iatom])
plt.xlabel('Molecular Configuration')

plt.plot(q,label='Mulliken charges from SIESTA',
         color='r', linewidth=2, linestyle='--')
plt.plot(q_,label='Qeq charges from ReaxFF',
         color='b', linewidth=2, linestyle=':')
plt.plot(qb,label='Qeq charges from ReaxFF(after train)',
         color='g', linewidth=2, linestyle=':')

plt.legend(loc='best',edgecolor='yellowgreen') # lower left
plt.savefig('charge.eps',transparent=True) 
plt.close()


d.close()
# doc.show()

