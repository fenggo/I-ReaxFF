#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
from irff.reaxfflib import read_lib,write_lib
from irff.reax import ReaxFF
import argh
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.io.trajectory import Trajectory
import tensorflow as tf
import json as js


colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']


def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        # key = key.encode('raw_unicode_escape')
        # print(key)
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           # print(kk)
           if len(kk)==2:
              offd.append(k[1])
           elif len(kk)==1:
              spec.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return spec,bonds,offd,angs,torp,hbs


def get_p(ffield):
    if ffield.endswith('.json'):
       lf = open(ffield,'r')
       j = js.load(lf)
       p  = j['p']
       m       = j['m']
       spec,bonds,offd,angs,torp,hbs= init_bonds(p)
    else:
       p,zpe_,spec,bonds,offd,Angs,torp,Hbs=read_lib(libfile=ffield,zpe=False)
    return p,bonds


def get_bo(r,rosi=1.3935,bo1=-0.075,bo2=5.0):
    bo   = np.exp(bo1*(r/rosi)**bo2)
    return bo


def pb(r,b,color,bd,lab):
    r   = np.arange(1.0,3.0,0.1)
    rosi1,bo11,bo21 = 1.3935,-0.075,5.0
    rosi2,bo12,bo22 = 1.3935,-0.005,10.0

    b1  = get_bo(r,rosi=rosi1,bo1=bo11,bo2=bo21)
    b2  = get_bo(r,rosi=rosi2,bo1=bo12,bo2=bo22)

    plt.figure()
    plt.ylabel('Bond Order Uncorrected')
    plt.xlabel('Radius')

    plt.plot(r,b1,label=r'$r_{si}=%f,bo_1=%f,bo_2=%f$' %(rosi1,bo11,bo21),
             color='b', linewidth=2, linestyle='--')
    plt.plot(r,b2,label=r'$r_{si}=%f,bo_1=%f,bo_2=%f$' %(rosi2,bo12,bo22),
             color='b', linewidth=2, linestyle='-.')

    plt.legend()
    plt.savefig('boc.eps') 
    plt.close()


def plbo(lab='si'):
    p,bonds = get_p('ffield.json')
    r = np.arange(0.0001,2.6,0.1)

    plt.figure()
    plt.ylabel( 'Uncorrected '+lab+'Bond Order (%s)' %lab)
    plt.xlabel(r'$Radius$ $(Angstrom)$')
    plt.xlim(0,2.5)
    plt.ylim(0,1.01)

    for i,bd in enumerate(bonds):
        b = bd.split('-')
        bdn = b[0] if b[0]==b[1] else bd 
        if lab=='si':
           print(bd,p['rosi_'+bdn],p['bo1_'+bd],p['bo2_'+bd])
           b=get_bo(r,rosi=p['rosi_'+bdn],bo1=p['bo1_'+bd],bo2=p['bo2_'+bd])
        elif lab=='pi':
           print(bd,p['ropi_'+bdn],p['bo3_'+bd],p['bo4_'+bd])
           b=get_bo(r,rosi=p['ropi_'+bdn],bo1=p['bo3_'+bd],bo2=p['bo4_'+bd])
        elif lab=='pp':
           print(bd,p['ropp_'+bdn],p['bo5_'+bd],p['bo6_'+bd])
           b=get_bo(r,rosi=p['ropp_'+bdn],bo1=p['bo5_'+bd],bo2=p['bo6_'+bd])

        plt.plot(r,b,label=r'$%s$' %bd, 
                 color=colors[i%len(colors)], linewidth=2, linestyle='--')

    plt.legend()
    plt.savefig('bo_%s.eps' %lab) 
    plt.close()
           

def plbo1(lab='sigma'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= \
           read_lib(libfile='ffield',zpe=False)
    r = np.arange(0.0001,3.0,0.1)
    bo1_= np.arange(-2.0,-0.0001,0.05)

    plt.figure()
    plt.ylabel( 'Uncorrected '+lab+'Bond Order')
    plt.xlabel(r'$Radius$ $(Angstrom)$')
    plt.xlim(0,2.6)
    plt.ylim(0,1.01)
 
    bd = 'C-C'
    # for i,bd in enumerate(bonds):

    b = bd.split('-')
    bdn = b[0] if b[0]==b[1] else bd 
 
    for i,bo1 in enumerate(bo1_):
        b=get_bo(r,rosi=p['rosi_'+bdn],bo1=bo1,bo2=p['bo2_'+bd])

        if i==0:
           plt.plot(r,b,label=r'$bo1=%f$' %bo1, 
                 color=colors[i%len(colors)], linewidth=2, linestyle='--')
        elif i==len(bo1_)-1:
           plt.plot(r,b,label=r'$bo1=%f$' %bo1, 
                 color=colors[i%len(colors)], linewidth=2, linestyle='--')
        else:
           plt.plot(r,b, # label=r'$bo1=%f$' %bo1, 
                 color=colors[i%len(colors)], linewidth=2, linestyle='--') 

    plt.legend()
    plt.savefig('bovsbo1.eps') 
    plt.close()


def plbo2(lab='sigma'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= \
           read_lib(libfile='ffield',zpe=False)
    r = np.arange(0.0001,3.0,0.1)
    bo2_= np.arange(0.001,18.0,2.0)

    plt.figure()
    plt.ylabel( 'Uncorrected '+lab+'Bond Order')
    plt.xlabel(r'$Radius$ $(Angstrom)$')
    plt.xlim(0,3.0)
    plt.ylim(0,1.01)
 
    bd = 'C-C'
    # for i,bd in enumerate(bonds):

    b = bd.split('-')
    bdn = b[0] if b[0]==b[1] else bd 
 
    for i,bo2 in enumerate(bo2_):
        b=get_bo(r,rosi=p['rosi_'+bdn],bo1=p['bo1_'+bd],bo2=bo2)

        plt.plot(r,b,label=r'$bo2=%f$' %bo2,  
                 color=colors[i%len(colors)], linewidth=2, linestyle='--')

    plt.legend()
    plt.savefig('bovsbo2.eps') 
    plt.close()


def plbo3d(lab='sigma'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= \
           read_lib(libfile='ffield',zpe=False)
    r = np.linspace(0.0001,3.0,50)
    bo2_= np.linspace(0.001,18.0,50)
    bo1_= np.linspace(-0.9,-0.0001,50)

    bo1_,bo2_ = np.meshgrid(bo1_,bo2_)
    bd = 'C-C'
    b = bd.split('-')
    bdn = b[0] if b[0]==b[1] else bd 
    b=get_bo(r,rosi=p['rosi_'+bdn],bo1=bo1_,bo2=bo2_)

    fig = plt.figure()
    ax  = Axes3D(fig)
    # plt.xlabel("Delta'")
    ax  = plt.subplot(111, projection='3d')
    ax.plot_surface(bo1_,bo2_,b,cmap=plt.get_cmap('rainbow'))
    ax.contourf(bo1_,bo2_,b,zdir='z', offset=0.0, cmap=plt.get_cmap('rainbow'))

    plt.savefig('bovsbo23d.eps') 
    plt.close()


def plro(lab='sigma'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= \
           read_lib(libfile='ffield',zpe=False)
    r = np.arange(0.0001,3.0,0.1)
    ro= np.arange(0.75,1.85,0.15)

    plt.figure()
    plt.ylabel( 'Uncorrected '+lab+'Bond Order')
    plt.xlabel(r'$Radius$ $(Angstrom)$')
    plt.xlim(0,3.0)
    plt.ylim(0,1.01)
 
    bd = 'C-H'
    # for i,bd in enumerate(bonds):

    b = bd.split('-')
    bdn = b[0] if b[0]==b[1] else bd 
 
    for i,rosi in enumerate(ro):
        b=get_bo(r,rosi=rosi,bo1=p['bo1_'+bd],bo2=p['bo2_'+bd])

        plt.plot(r,b,label=r'$r_{sigma}=%f$' %rosi, 
                 color=colors[i%len(colors)], linewidth=2, linestyle='--')

    plt.legend()
    plt.savefig('bovsro.eps') 
    plt.close()


def plot_time(logs=None):
    plt.figure()          
    plt.ylabel('Elapsed Time (CPU Time)')
    plt.xlabel('Train step')
    c = 0
    for i,log in enumerate(logs):
        log1 = log.split('.')[0]
        log2 = log1.split('_')[1]
        fl = open(log,'r')
        tim = []
        spe = []
        t = 0.0
        for il,line in enumerate(fl.readlines()):
            if il<9000:
               l = line.split() 
               if len(l)>2:
                  if l[1]=='step:':
                     t += float(l[-1])
                     tim.append(t)
        # plt.plot(spe,color='red',alpha=0.2,label='Sum of square error')
        plt.plot(tim,color=colors[i%len(colors)],alpha=0.2,label=log2)
    plt.legend(loc='best')
    plt.savefig('time_usage.eps') 
    plt.close() 


def plot_log(log=None):
    spe,accs = [],{}
    fl = open(log,'r')
    lines = fl.readlines()
    fl.close()

    line = lines[30].split()
    tim  = []
    for i,l in enumerate(line):
        if i>=7 and i<len(line)-2:
           if i%2==1:
              key = l[:-1]
              # print(key)
              accs[key] = []

    # for log in logs:
    fl = open(log,'r')
    for line in fl.readlines():
        if line.find('nan')>=0 :
           continue
        if line.find('-  step:')>=0 :
           l = line.split() 
           spe.append(float(l[4]))
           for i,s in enumerate(l):
               if i>=7 and i<len(l)-2:
                  if i%2==1:
                     if s[:-1]!='spv':
                        accs[s[:-1]].append(float(l[i+1]))
                     tim.append(float(l[-1]))
    
    plt.figure()             # temperature
    plt.ylabel('Sum of square error (eV)')
    plt.xlabel('Train step x100')
    plt.plot(spe,color='red',alpha=0.2,label='Sum of square error')
    plt.legend(loc='best')
    plt.savefig('sqe.eps') 
    plt.close() 

    plt.figure() 
    ploted = []
    for i,key in enumerate(accs):  
        k = key.split('-')[0]
        if k not in ploted:
           plt.ylabel('Accuracy')
           plt.xlabel('train step x100')
           plt.ylim(0.0,1.0)
           plt.plot(accs[key],color=colors[i%len(colors)],alpha=0.2,label=k)
           ploted.append(k)
    plt.legend(loc='best')
    plt.savefig('acc.eps') 
    plt.close() 


def plot_delta(direcs=None,batch_size=1,dft='siesta'):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile='ffield',direcs=direcs,dft=dft,
                 opt=[],optword='all',
                 batch_size=batch_size,
                 rc_scale='none',
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')
    D     = rn.get_value(rn.D)
    Dlp   = rn.get_value(rn.Dlp)
 
    Dp_ = {}
    for sp in rn.spec:
        if rn.nsp[sp]>0:
           Dp_[sp] = tf.gather_nd(rn.Deltap,rn.atomlist[sp])
    Dp   = rn.get_value(Dp_)

    atlab = rn.lk.atlab

    traj  = Trajectory('delta.traj','w')
    trajp = Trajectory('deltap.traj','w')
    trajlp= Trajectory('deltalp.traj','w')

    natom = molecules[mol].natom
    d     = np.zeros([natom,batch_size])
    dp    = np.zeros([natom,batch_size])
    dlp   = np.zeros([natom,batch_size])
    cell  = rn.cell[mol]

    for sp in rn.spec:
        if rn.nsp[sp]>0:
           for l,lab in enumerate(atlab[sp]):
               if lab[0]==mol:
                  i = int(lab[1])
                  d[i]  = D[sp][l]
                  dp[i] = Dp[sp][l]
                  dlp[i]= Dlp[sp][l]

    for nf in range(batch_size):
        A = Atoms(symbols=molecules[mol].atom_name,
                  positions=molecules[mol].x[nf],
                  charges=d[:,nf],
                  cell=cell,
                  pbc=(1, 1, 1))
        traj.write(A)

        Ap = Atoms(symbols=molecules[mol].atom_name,
                  positions=molecules[mol].x[nf],
                  charges=dp[:,nf],
                  cell=cell,
                  pbc=(1, 1, 1))
        trajp.write(Ap)

        Alp= Atoms(symbols=molecules[mol].atom_name,
                  positions=molecules[mol].x[nf],
                  charges=dlp[:,nf],
                  cell=cell,
                  pbc=(1, 1, 1))
        trajlp.write(Alp)
    traj.close()
    trajp.close()
    trajlp.close()


def plddlp(direcs={'ethane':'/home/gfeng/siesta/train/ethane'},
         val='bo2_C-C',batch_size=1000):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile='ffield',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 rc_scale='none',
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    sp      = 'O'
    dlp     = rn.get_value(rn.Delta_lp)


    # d[natom,nframe] dp[natom,nframe]
    plt.figure()             # temperature
    plt.ylabel(r'$\Delta$ distribution')
    plt.xlabel(r"The value of $\Delta$ and $\Delta'$")
    n1, bins1, patches1 = plt.hist(dlp[sp],bins=20,facecolor='none',edgecolor='blue',
                                   normed=1,histtype='step',alpha=0.5,label=r"$\Delta_{lp}(O)$")

    plt.legend(loc='best')
    plt.savefig('deltalp_distribution.eps') 
    plt.close() 


def plddd(direcs={'ethane':'/home/gfeng/siesta/train/ethane'},
         spec='C',nbin=500,
         batch_size=2000):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile='ffield',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')


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


    plt.figure()             # temperature
    plt.ylabel(r'$\Delta$ distribution')
    plt.xlabel(r"The value of $\Delta$ and $\Delta'$")
    
    hist,bin_=np.histogram(Dp[spec],range=(np.min(Dp[spec]),np.max(Dp[spec])),bins=nbin,density=True)
    plt.plot(bin_[:-1],hist,alpha=0.5,color='blue',
             linestyle=':',label=r"$\Delta^{'}(%s)$ " %spec)
    
    histp,bin_=np.histogram(D[spec],range=(np.min(D[spec]),np.max(D[spec])),bins=nbin,density=True)
    plt.plot(bin_[:-1],histp,alpha=0.5,color='yellowgreen',
             linestyle='-.',label=r'$\Delta$(%s)' %spec)

    plt.legend(loc='best')
    plt.savefig('delta_%s.eps' %spec)  
    plt.close() 


def get_f4(boc3=0.51,boc4=8.4358,boc5=0.3687,D=-0.12):
    Di_boc = D
    f4r = np.exp(-boc3*(boc4*np.square(0.28)-Di_boc)+boc5)
    f4  = 1.0/(1.0+f4r)
    return f4


def plf4():
    Delta = np.arange(2.0,3.0,0.01)
    D1 = Delta- 3.2291 
    D2 = Delta- 3.2291 
    # Delta1,Delta2  = np.meshgrid(Delta1, Delta2)
    D1,D2  = np.meshgrid(D1, D2)

    boc4,boc3,boc5 = 12.4973,7.0211,8.9499
    f4  = get_f4(boc3=boc3,boc4=boc4,D=D1)
    f5  = get_f4(boc3=boc3,boc4=boc4,D=D2)

    f45 = f4*f5

    fig = plt.figure()
    ax  = Axes3D(fig)

    ax  = plt.subplot(111, projection='3d')
    ax.plot_surface(D1,D2,f45, cmap=plt.get_cmap('rainbow'))
    ax.contourf(D1,D2,f45, zdir='z', offset=0.9994, cmap=plt.get_cmap('rainbow'))

    plt.savefig('f4.eps') 
    plt.close()


def get_f2(boc1,Di,Dj):
    return np.exp(-boc1*Di)+np.exp(-boc1*Dj)


def get_f3(boc2,Di,Dj):
    boc2_ = -1.0/boc2
    return boc2_*np.log(0.5*np.exp(boc2*Di)+0.5*np.exp(boc2*Dj))


def get_f1(vali,valj,f2,f3):
    return 0.5*(vali+f2)/(vali+f2+f3)+0.5*(valj+f2)/(valj+f2+f3)


def plf2():
    boc1,boc2 = 7.5772 , 4.8418
    val    = 3.2291
    Delta1 = np.arange(2.00,3.0,0.01)
    D1     = Delta1- 3.2291 

    Delta2 = np.arange(2.0,3.0,0.01)
    D2     = Delta2- 3.2291 
    
    Delta1,Delta2  = np.meshgrid(Delta1, Delta2)
    D1,D2  = np.meshgrid(D1, D2)
    f2     = get_f2(boc1,D1,D2)

    fig = plt.figure()
    ax  = Axes3D(fig)
    ax  = plt.subplot(111, projection='3d')

    ax.plot_surface(Delta1,Delta2,f2, cmap=plt.get_cmap('rainbow'))
    # ax.contourf(Delta1,Delta2,f2, zdir='z', offset=0.0, cmap=plt.get_cmap('rainbow'))

    plt.savefig('f2.eps') 
    plt.close()


def plf3():
    boc1,boc2 = 7.5772 , 4.8418
    val    = 3.2291
    Delta1 = np.arange(2.00,3.0,0.01)
    D1     = Delta1- 3.2291 

    Delta2 = np.arange(2.0,3.0,0.01)
    D2     = Delta2- 3.2291 
    
    Delta1,Delta2  = np.meshgrid(Delta1, Delta2)
    D1,D2  = np.meshgrid(D1, D2)
    f3     = get_f3(boc2,D1,D2)

    fig = plt.figure()
    ax  = Axes3D(fig)
    ax  = plt.subplot(111, projection='3d')
    ax.view_init(elev=10,azim=20)

    ax.plot_surface(Delta1,Delta2,f3, cmap=plt.get_cmap('rainbow'))
    # ax.contourf(Delta1,Delta2,f3, zdir='z', offset=0.2, cmap=plt.get_cmap('rainbow'))

    plt.savefig('f3.eps') 
    plt.close()


def plf1d3():
    boc1,boc2 = 7.5772 , 4.8418
    val    = 3.2291
    Delta1 = np.arange(2.00,3.0,0.01)
    D1     = Delta1- 3.2291 

    Delta2 = np.arange(2.0,3.0,0.01)
    D2     = Delta2- 3.2291 
    
    Delta1,Delta2  = np.meshgrid(Delta1, Delta2)
    D1,D2  = np.meshgrid(D1, D2)
    f2     = get_f2(boc1,D1,D2)
    f3     = get_f3(boc2,D1,D2)
    f1     = get_f1(val,val,f2,f3)
    # f1 = f1*100

    fig = plt.figure()
    ax  = Axes3D(fig)
    # plt.xlabel("Delta'")
    ax  = plt.subplot(111, projection='3d')
    ax.plot_surface(Delta1,Delta2,f1, cmap=plt.get_cmap('rainbow'))
    ax.contourf(Delta1,Delta2,f1, zdir='z', offset=0.984, cmap=plt.get_cmap('rainbow'))

    # plt.show()
    # plt.plot(Delta,f4,label=r"f4 as a function of Delta'",
    #          color='b', linewidth=2, linestyle='-')
    # plt.legend()

    plt.savefig('f1.eps') 
    plt.close()


def plf1(direcs={'ethane':'/home/gfeng/siesta/train/ethane'},batch_size=1000):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile='ffield',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 rc_scale='none',
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    bd  = 'C-C'
    bd1 = 'C-H'
    bd2 = 'H-O'
    f_1 = rn.get_value(rn.f_1)
    f_2 = rn.get_value(rn.f_2)
    f_3 = rn.get_value(rn.f_3)



    # d[natom,nframe] dp[natom,nframe]
    plt.figure()             # temperature
    plt.ylabel(r'f1 distribution')
    plt.xlabel('The value of f1')
    plt.hist(f_1[bd],bins=32,facecolor='none',edgecolor='blue',
                                   normed=1,histtype='step',alpha=0.5,label=r'$f1(C-C)$')
    
    plt.hist(f_1[bd1],bins=32,facecolor='none',edgecolor='yellowgreen',
                                     normed=1,histtype='step',alpha=0.5,label=r'$f1(C-H)$')

    plt.hist(f_1[bd2],bins=32,facecolor='none',edgecolor='red',
                                     normed=1,histtype='step',alpha=0.5,label=r'$f1(O-H)$')

    plt.legend(loc='best')
    plt.savefig('f1_distribution.eps') 
    plt.close() 



if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [pb,plbo3d,plbo,plro,plbo1,plbo2,plddlp,
   	                          plf4,plf1,plf1d3,plf2,plf3])
   argh.dispatch(parser)

