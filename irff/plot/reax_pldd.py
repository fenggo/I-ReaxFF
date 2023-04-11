from __future__ import print_function
from irff.reax import ReaxFF
import numpy as np
import matplotlib.pyplot as plt



def pldd(direcs={'ethane':'/home/gfeng/siesta/train/ethane'},
         batch_size=50,
         atoms=[8,51]):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile='ffield',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    atom_name = molecules[mol].atom_name
    delta  = []
    deltap = []

    D     = rn.get_value(rn.D)
    atlab = rn.lk.atlab
    natom = molecules[mol].natom
    d     = np.zeros([natom,batch_size])
    dp    = np.zeros([natom,batch_size])
    cell  = rn.cell[mol]

    Dp_ = {}
    for sp in rn.spec:
        if rn.nsp[sp]>0:
           Dp_[sp] = tf.gather_nd(rn.Deltap,rn.atomlist[sp])
    Dp = rn.get_value(Dp_)

    for sp in rn.spec:
        if rn.nsp[sp]>0:
           for l,lab in enumerate(atlab[sp]):
               if lab[0]==mol:
                  i = int(lab[1])
                  d[i] = D[sp][l]
                  dp[i] = Dp[sp][l]

    plt.figure()      
    plt.ylabel(r'$\Delta$ distribution')
    plt.xlabel(r"The value of $\Delta$ and $\Delta'$")
    # plt.xlim(0.01,3.0)
    # plt.ylim(0,50)
    
    for i,atm in enumerate(atoms):
        plt.plot(dp[atm],alpha=0.5,color=colors[(i*2)%len(colors)],
                 label=r"$\Delta^{'}$ of atom@%s:%d" %(atom_name[atm],atm))

        plt.plot(d[atm],alpha=0.5,color=colors[(i*2+1)%len(colors)],
                 label=r'$\Delta$ of atom@%s:%d' %(atom_name[atm],atm))

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('delta.eps',transparent=True) 
    plt.close() 

