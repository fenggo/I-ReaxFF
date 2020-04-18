#!/usr/bin/env python
from .mdtodata import MDtoData
from os import getcwd
from ase.io.trajectory import TrajectoryWriter,Trajectory
from math import ceil
import numpy as np


def prep_data(label=None,direcs=None,split_batch=100,frame=50,max_batch=50):
    ''' To sort data must have same atom number and atom types 
          images: contains all atom image in all directions
          frame : collect this number to images
          split_batch: split images evergy split_batch
        In the end, all data stored in label-i.traj file:
          collect number (frame=5) in energy directory, split the frame to traj file 
          evergy (split_batch=1000)
    '''
    images = []
    for key in direcs:
        try:
           d = MDtoData(structure=key,dft='siesta',direc=direcs[key],batch=frame)
        except:
           print('-  An error occurred with case %s, neglected.' %key)

        images_ = d.get_images()
        # print('-  number of frames:',len(images_))
        if len(images_)>frame:
           images.extend(images_[0:frame])
        else:
           images.extend(images_)
        d.close()

    # traj = TrajectoryWriter('all.traj',mode='w')
    # for atoms in images:
    #     traj.write(atoms=atoms)

    nframe = len(images)                        # get batch size to split
    if nframe>split_batch :                            
       nb_    = float(int(nframe/split_batch))
       spb_   = int(ceil(nframe/nb_))
       if nb_>max_batch:
          nb_ = max_batch
          spb_= int(ceil(nframe/max_batch))
    else:
       nb_    = 1
       spb_   = split_batch   
       
    n = int(nb_)
    if n*split_batch<nframe:
       pool   = np.array(nframe)
       ind_   = np.linspace(0,nframe-1,num=n*split_batch,dtype=np.int32)
       images = [images[_] for _ in ind_]

    trajs = {}
    for i in range(n):
        sf = i*split_batch
        ef = (i+1)*split_batch
        if sf<nframe:
           if ef>nframe:
              ef = nframe
           # print(i,sf,ef)
           images_ = images[sf:ef]
           tn      = label+'-'+str(i)
           traj    = TrajectoryWriter(tn+'.traj',mode='w')
           for atoms in images_:
               traj.write(atoms=atoms)
           traj.close()
           trajs[tn] = tn+'.traj'
    return trajs


if __name__ == '__main__':
   ''' the atom number and atom types must be same 
   '''
   direcs={'hb6':'/media/feng/NETAC/siesta/hb6',
           'hb10':'/media/feng/NETAC/siesta/hb10',
           'hb11':'/media/feng/NETAC/siesta/hb11',
           'hb12':'/media/feng/NETAC/siesta/hb12',
            }

   prep_data(label='hb',direcs=direcs,split_batch=100,frame=50)

