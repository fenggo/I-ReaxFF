#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
from irff.irff_np import IRFF_NP
from irff.AtomOP import AtomOP
import argh
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.io import read
import json as js


colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','agua','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']

def vdwtaper(r,vdwcut=10.0):
    tp = 1.0+np.divide(-35.0,np.power(vdwcut,4.0))*np.power(r,4.0)+ \
         np.divide(84.0,np.power(vdwcut,5.0))*np.power(r,5.0)+ \
         np.divide(-70.0,np.power(vdwcut,6.0))*np.power(r,6.0)+ \
         np.divide(20.0,np.power(vdwcut,7.0))*np.power(r,7.0)
    return tp


def f13(r,ai='N',aj='N'):
    with open('ffield.json','r') as lf:
         j = js.load(lf)
    p  = j['p']
    gw = np.sqrt(p['gammaw_'+ai]*p['gammaw_'+aj])
    print(gw)
    rr = np.power(r,p['vdw1'])+np.power(np.divide(1.0,gw),p['vdw1'])
    f  = np.power(rr,np.divide(1.0,p['vdw1']))  
    return f


def pltp():
    r = np.linspace(0.2,10.0,100)

    tp = vdwtaper(r)

    fig, ax = plt.subplots() 
    plt.xlabel(r'$Radius$')
    plt.ylabel(r'$Taper$ $Function$')
    plt.plot(r,tp,label=r'$Burchart$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('vdwtaper.pdf') 
    plt.close()


def plf13():
    r = np.linspace(0.01,3.0,100)
    f = f13(r)

    fig, ax = plt.subplots() 
    plt.xlabel(r'$Radius$')
    plt.ylabel(r'$f_{13}$ $Function$')
    plt.plot(r,f,label=r'$f_{13}$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('f13.pdf') 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   # pltp()
   plf13()

