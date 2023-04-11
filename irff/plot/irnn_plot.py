#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
from irff.irnnlib import read_lib,write_lib
from irff.irnn import IRNN
from irff.sigmoid import plot_s,sigmoid 
import argh
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.io.trajectory import Trajectory
import tensorflow as tf


colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','agua','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']


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


def plbo(lab='sigma'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= \
           read_lib(libfile='ffield',zpe=False)
    r = np.arange(0.0001,3.0,0.1)

    plt.figure()
    plt.ylabel( 'Uncorrected '+lab+'Bond Order')
    plt.xlabel(r'$Radius$ $(Angstrom)$')
    # plt.xlim(0,2.5)
    # plt.ylim(0,1.01)

    for i,bd in enumerate(bonds):
        b = bd.split('-')
        bdn = b[0] if b[0]==b[1] else bd 
        if lab=='sigma':
           bo_=get_bo(r,rosi=p['rosi_'+bdn],bo1=p['bo1_'+bd],bo2=p['bo2_'+bd])
           bo = sigmoid(p['bosiw1_'+bd]*bo_+p['bosib1_'+bd])
        elif lab=='pi':
           bo_=get_bo(r,rosi=p['ropi_'+bdn],bo1=p['bo3_'+bd],bo2=p['bo4_'+bd])
           bo = sigmoid(bo_*p['bopiw1_'+bd]+p['bopib1_'+bd])
        elif lab=='pp':
           bo_=get_bo(r,rosi=p['ropp_'+bdn],bo1=p['bo5_'+bd],bo2=p['bo6_'+bd])
           bo = sigmoid(bo_*p['boppw1_'+bd]+p['boppb1_'+bd])

        plt.plot(r,bo,label=r'$%s$' %bd, 
                 color=colors[i%len(colors)], linewidth=2, linestyle='--')

    plt.legend()
    plt.savefig('bo_%s.eps' %lab) 
    plt.close()
           

def plbo1(lab='sigma'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= \
           read_lib(libfile='ffield',zpe=False)
    r = np.arange(0.0001,3.0,0.1)
    bo1_= np.arange(-0.9,-0.0001,0.1)

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

        plt.plot(r,b,label=r'$bo1=%f$' %bo1, 
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
 
    bd = 'C-C'
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
    # blue green red cyan magenta black white
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
    rn = IRNN(libfile='ffield.json',direcs=direcs,dft=dft,
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


def pldlc(direcs={'ch4c-1':'/home/feng/siesta/train2/ch4c1'},
         val='bo2_C-C',batch_size=200):
    for m in direcs:
        mol = m
    rn = IRNN(libfile='ffield.json',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 rc_scale='none',
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    sp      = 'O'
    dlc     = rn.get_value(rn.Delta_lpcorr)
    dlp     = rn.get_value(rn.Delta_lp)   

    # d[natom,nframe] dp[natom,nframe]
    plt.figure()             # temperature
    plt.ylabel(r'$\Delta$ distribution')
    plt.xlabel(r"The value of $\Delta$ and $\Delta'$")

    nb_ = 500
    sp  = 'C'
    hist,bin_ = np.histogram(dlp[sp],range=(-3.0,3.0),bins=nb_,density=True)
    plt.plot(bin_[:-1],hist,alpha=0.5,color='blue',label=r"$\Delta^{lp}(%s)$ " %sp)
    
    hist,bin_ = np.histogram(dlc[sp],range=(-3.0,3.0),bins=nb_,density=True)
    plt.plot(bin_[:-1],hist,alpha=0.5,color='yellowgreen',label=r'$\Delta^{lc}$(%s)' %sp)

    plt.legend(loc='best')
    plt.savefig('deltalc_%s.eps' %sp)  
    plt.close() 



def pldf(direcs={'ethane':'/home/feng/siesta/train2/ethane'},
         val='bo2_C-C',batch_size=1000):
    for m in direcs:
        mol = m
    rn = IRNN(libfile='ffield.json',direcs=direcs,dft='siesta',
              optword='all',
              batch_size=batch_size,
              rc_scale='none',
              clip_op=False,
              interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')
    
    df4     = rn.get_value(rn.df4)
    bonds   = rn.bonds
    bd      = 'C-H'

    plt.figure()             # temperature
    plt.ylabel(r'$\Delta$ distribution')
    plt.xlabel(r"The value of $\Delta$ minus bo")

    nb_ = 100
    hist,bin_ = np.histogram(df4[bd],range=(-4.0,1.0),bins=nb_,density=True)
    plt.plot(bin_[:-1],hist,alpha=0.5,color='blue',label=r"$\Delta^{'}$")

    plt.legend(loc='best')
    plt.savefig('delta_minus_bo.eps') 
    plt.close() 


def pldd(direcs={'ethane':'/home/feng/siesta/ethane'},
         val='bo2_C-C',batch_size=1000):
    for m in direcs:
        mol = m
    rn = IRNN(libfile='ffield.json',direcs=direcs,dft='siesta',
              optword='all',
              batch_size=batch_size,
              rc_scale='none',
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

    # d[natom,nframe] dp[natom,nframe]
    plt.figure()             # temperature
    plt.ylabel(r'$\Delta$ distribution')
    plt.xlabel(r"The value of $\Delta$ and $\Delta'$")


    nb_ = 500
    sp  = 'O'
    hist,bin_=np.histogram(Dp[sp],range=(np.min(Dp[sp]),np.max(Dp[sp])),bins=nb_,density=True)
    plt.plot(bin_[:-1],hist,alpha=0.5,color='blue',
             linestyle=':',label=r"$\Delta^{'}(%s)$ " %sp)
    
    histp,bin_=np.histogram(D[sp],range=(np.min(D[sp]),np.max(D[sp])),bins=nb_,density=True)
    plt.plot(bin_[:-1],histp,alpha=0.5,color='yellowgreen',
             linestyle='-.',label=r'$\Delta$(%s)' %sp)

    plt.legend(loc='best')
    plt.savefig('delta_%s.eps' %sp)  
    plt.close() 


def delta(direcs={'ethane':'/home/gfeng/siesta/train/ethane'},
          val='bo2_C-C',batch_size=1):
    for m in direcs:
        mol = m
    rn = IRNN(libfile='ffield.json',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 rc_scale='none',
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')


    delta  = []
    deltap = []
    xs     = np.linspace(0.9,20.0,50)
    atom   = 18

    for x in xs:
        rn.sess.run(tf.assign(rn.v[val],x))

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
        Dp   = rn.get_value(Dp_)


        for sp in rn.spec:
            if rn.nsp[sp]>0:
               for l,lab in enumerate(atlab[sp]):
                   if lab[0]==mol:
                      i = int(lab[1])
                      d[i] = D[sp][l]
                      dp[i] = Dp[sp][l]
        delta.append(d[atom][0])
        deltap.append(dp[atom][0])

    # d[natom,nframe] dp[natom,nframe]
    plt.figure()             # temperature
    plt.ylabel('Delta')
    plt.xlabel(val)
    plt.plot(xs,delta,color='red',alpha=0.2,label=r'Delta')
    plt.plot(xs,deltap,color='blue',alpha=0.2,label='Delta\'')
    plt.legend(loc='best')
    plt.savefig('delta.eps') 
    plt.close() 


def get_f4(boc3=0.51,boc4=8.4358,D=-0.12):
    df4 = D
    f4r = np.exp(-boc3*df4+boc4)
    f4  = 1.0/(1.0+f4r)
    return f4


def plf4():
    '''  get parameters from ffield  '''
    p,zpe,spec,bonds,offd,angs,torp,hbs= \
           read_lib(libfile='ffield',zpe=False)
    df4 = np.linspace(-2.500,-1.0,100)

    plt.figure()
    plt.ylabel( 'f4')
    plt.xlabel(r'bo - $\Delta$')

    for i,bd in enumerate(bonds):
        f4_= get_f4(boc3=p['boc3_'+bd],boc4=p['boc4_'+bd],D=df4)
        f4 = sigmoid(p['f4w1_'+bd]*f4_+p['f4b1_'+bd])

        plt.plot(df4,f4,label=r'$%s$' %bd, 
                 color=colors[i%len(colors)], linewidth=2, linestyle='--')

    plt.legend()
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
    boc1,boc2 = 1.5772 , 1.8418
    val    = 1.2291
    Delta1 = np.arange(1.00,2.5,0.01)
    D1     = Delta1- 3.2291 

    Delta2 = np.arange(1.0,2.5,0.01)
    D2     = Delta2- 3.2291 
    
    Delta1,Delta2  = np.meshgrid(Delta1, Delta2)
    D1,D2  = np.meshgrid(D1, D2)
    f2     = get_f2(boc1,D1,D2)
    f3     = get_f3(boc2,D1,D2)
    f1_    = get_f1(val,val,f2,f3)
    f1     = sigmoid(-0.5*f1_+0.5)
    f1     = sigmoid(-0.5*f1+0.5)
    # f1 = f1*100

    fig = plt.figure()
    ax  = Axes3D(fig)
    # plt.xlabel("Delta'")
    ax  = plt.subplot(111, projection='3d')
    ax.plot_surface(Delta1,Delta2,f1, cmap=plt.get_cmap('rainbow'))
    ax.contourf(Delta1,Delta2,f1, zdir='z', offset=0.91, cmap=plt.get_cmap('rainbow'))

    plt.savefig('f1.eps') 
    plt.close()



def s():
    plot_s()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [pb,plbo3d,plbo,plro,plbo1,plbo2,delta,pldd,pldlc,
   	                          pldf,plf4,plf1d3,plf2,plf3,s])
   argh.dispatch(parser)

