import numpy as np
from ..irff_np import IRFF_NP
import matplotlib.pyplot as plt


def theta(images):
    atoms = images[0]
    ir = IRFF_NP(atoms=atoms,
                libfile='ffield.json',
                nn=True)
    ir.calculate(atoms)

    for a,ang in enumerate(ir.angs):
        i_,j_,k_ = ang
        if ir.eang[a]>0.00000001:
           print('{:3d} {:2d}-{:2d}-{:2d}  {:s}-{:s}-{:s}  {:8.4f} fijk {:8.4f} f7 {:8.4f} f8 {:8.4f}'.format(a,i_,j_,k_,
                 ir.atom_name[i_],ir.atom_name[j_],ir.atom_name[k_],ir.eang[a],
                 ir.fijk[a],ir.f_7[a],ir.f_8[a]))
        elif ir.eang[a]<0.0:
           print('{:3d} {:2d}-{:2d}-{:2d}  {:s}-{:s}-{:s}  {:8.4f} fijk {:8.4f} f7 {:8.4f} f8 {:8.4f}'.format(a,i_,j_,k_,
                 ir.atom_name[i_],ir.atom_name[j_],ir.atom_name[k_],ir.eang[a],
                 ir.fijk[a],ir.f_7[a],ir.f_8[a]))
            

    a_ = int(input('please input the id of the  angle to output(-1 to exit): '))
    i,j,k = ir.angs[a_]
    ang_ =  '{:s}-{:s}-{:s}'.format(ir.atom_name[i],ir.atom_name[j],ir.atom_name[k])

    theta_ = []
    Eang   = []

    if a_>=0:
        for atoms in images:
            ir.calculate(atoms)
            a = np.where(np.logical_and(np.logical_and(ir.angs[:,0]==i,ir.angs[:,1]==j),ir.angs[:,2]==k))
            a = np.squeeze(a)
             
            if len(ir.angs)==1:
               print('{:3d} theta0:{:6.4f} theta{:6.4f} Dang:{:4.2f} D:{:4.2f} rnlp:{:4.2f} '
                        'SBO:{:4.2f} sbo:{:4.2f} pbo:{:4.2f} SBO3:{:4.2f} Eang:{:6.4f}'.format(t,
                        ir.thet0[a],ir.theta,ir.dang,ir.Deltap[j],ir.rnlp,ir.SBO,ir.sbo,ir.pbo,
                        ir.SBO3,ir.eang[a])) # self.thet0-self.theta
               theta_.append(ir.theta)
               Eang.append(ir.eang[a])
            else:
               print('{:3d} t0:{:6.4f} t:{:6.4f} Dang:{:4.2f} D:{:4.2f} Nlp:{:4.2f} '
                     'S:{:4.2f} s:{:4.2f} pbo:{:4.2f} S3:{:4.2f} Eang:{:6.4f}'.format(t,
                     ir.thet0[a],ir.theta[a],ir.dang[a],ir.Deltap[j],ir.rnlp[a],ir.SBO[a],ir.sbo[a],ir.pbo[a],
                     ir.SBO3[a],ir.eang[a])) # self.thet0-self.theta
               theta_.append(ir.theta[a])
               Eang.append(ir.eang[a])
               # else:
               #    print(a)

        plt.figure()     
        # plt.plot(theta_,Eang,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$Eangle({:s})$'.format(ang_))
        plt.scatter(theta_,Eang,marker='o',color='none',edgecolors='r',s=10,label=r'$Eangle({:s})$'.format(ang_))

        plt.legend(loc='best',edgecolor='yellowgreen')
        plt.show() # if show else plt.savefig('deb_bo.pdf')
        plt.close()


