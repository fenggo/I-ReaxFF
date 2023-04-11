from __future__ import print_function
from irff.reax import ReaxFF
import numpy as np
import matplotlib.pyplot as plt



def plbd(direcs={'ethane':'/home/gfeng/siesta/train/ethane'},
         batch_size=50,
         nn=False,
         ffield='ffield',
         bonds=[9,41]):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile=ffield,direcs=direcs,dft='siesta',
                 optword='all',
                 nn=nn,
                 InitCheck=False,
                 batch_size=batch_size,
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    rbd    = rn.get_value(rn.rbd)
    bop    = rn.get_value(rn.bop)
    bo     = rn.get_value(rn.bo0)

    bopow1 = rn.get_value(rn.bopow1)
    eterm1 = rn.get_value(rn.eterm1)
    
    bop_si = rn.get_value(rn.bop_si)
    bosi   = rn.get_value(rn.bosi)
    sieng  = rn.get_value(rn.sieng)
    powb   = rn.get_value(rn.powb)
    expb   = rn.get_value(rn.expb)

    bop_pi = rn.get_value(rn.bop_pi)
    bopi   = rn.get_value(rn.bopi)

    bop_pp = rn.get_value(rn.bop_pp)
    bopp   = rn.get_value(rn.bopp)
 
    if not nn:
       f11    = rn.get_value(rn.F_11)
       f12    = rn.get_value(rn.F_12)
       f45    = rn.get_value(rn.F_45)
    f      = rn.get_value(rn.F)

    bdlab  = rn.lk.bdlab
    atom_name = molecules[mol].atom_name
    rn.close()
    
    bd=bonds
    bdn = atom_name[bd[0]]+'-'+atom_name[bd[1]]
    if not bdn in rn.bonds:
       bdn = atom_name[bd[1]]+'-'+atom_name[bd[0]]
    bn = [mol,bd[0],bd[1]]

    if bn in bdlab[bdn]:
       bid = bdlab[bdn].index(bn)
    else:
       bid = bdlab[bdn].index([mol,bd[1],bd[0]])

    plt.figure()    

    plt.subplot(3,2,1)        
    plt.plot(rbd[bdn][bid],alpha=0.5,color='b',
             linestyle='-',label="radius@%d-%d" %(bd[0],bd[1]))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,2)
    l = len(f[bdn][bid])
    plt.plot(f[bdn][bid],alpha=0.5,color='r',
                   linestyle='-',label="F@%d-%d" %(bd[0],bd[1]))
    if nn:
       plt.legend(loc='best',edgecolor='yellowgreen')
    else:
       x_ = int(0.3*l)
       y_ = f[bdn][bid][x_]
       yt_= y_ + 0.3 if y_<0.5 else y_ - 0.3
       plt.annotate('F', xy=(x_,y_), xycoords='data',
                     xytext=(x_, yt_),
                     arrowprops=dict(arrowstyle='->',facecolor='red'))
    
    if not nn:
       plt.plot(f11[bdn][bid],alpha=0.5,color='g',
                     linestyle='-.',label="F1@%d-%d" %(bd[0],bd[1]))
       x_ = int(0.6*l)
       y_ = f11[bdn][bid][x_]
       yt_= y_ + 0.3 if y_<0.5 else y_ - 0.3
       plt.annotate('F1', xy=(x_,y_), xycoords='data',
                     xytext=(x_, yt_),
                     arrowprops=dict(arrowstyle='->',facecolor='red'))

       plt.plot(f45[bdn][bid],alpha=0.5,color='b',
                      linestyle=':',label="F4@%d-%d" %(bd[0],bd[1]))
       x_ = int(0.9*l)
       y_ = f45[bdn][bid][x_]
       yt_= y_ + 0.3 if y_<0.5 else y_ - 0.3
       plt.annotate('F4', xy=(x_,y_), xycoords='data',
                     xytext=(x_, yt_),
                     arrowprops=dict(arrowstyle='->',facecolor='red'))

    plt.subplot(3,2,3)     
    plt.plot(bo[bdn][bid],alpha=0.5,color='b',
             linestyle='-',label="BO@%d-%d" %(bd[0],bd[1]))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,4)     
    plt.plot(bop_si[bdn][bid],alpha=0.5,color='b',
             linestyle=':',label=r"$BO^{'}_{si}$@%d-%d" %(bd[0],bd[1]))
    plt.plot(bosi[bdn][bid],alpha=0.5,color='r',
             linestyle='-',label=r'$BO_{si}$@%d-%d' %(bd[0],bd[1]))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,5)
    plt.plot(sieng[bdn][bid],alpha=0.5,color='b',
             linestyle='-',label=r"$ebond_{si}$@%d-%d" %(bd[0],bd[1]))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,6)
    plt.plot(powb[bdn][bid],alpha=0.5,color='b',
             linestyle='-',label=r"$pow_{si}$@%d-%d" %(bd[0],bd[1]))
    plt.plot(expb[bdn][bid],alpha=0.5,color='r',
             linestyle='-',label=r"$exp_{si}$@%d-%d" %(bd[0],bd[1]))
    plt.legend(loc='best',edgecolor='yellowgreen')

    # plt.subplot(3,2,6)
    # # plt.ylabel(r'$exp$ (eV)')
    # plt.xlabel(r"Step")
    # plt.plot(eterm1[bdn][bid],alpha=0.5,color='b',
    #          linestyle='-',label=r"$exp_{si}$@%d-%d" %(bd[0],bd[1]))
    # plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('bondorder.eps',transparent=True)  
    plt.close() 
    
    # for i in range(batch_size):
    #     print('-  F11: %f  F12: %f  F45: %f' %(f11[bdn][bid][i],f12[bdn][bid][i],f45[bdn][bid][i]))

