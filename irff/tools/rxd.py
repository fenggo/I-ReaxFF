#!/usr/bin/env python
from __future__ import print_function
from os import system, getcwd, chdir,listdir,environ
from os.path import isfile,exists,isdir
from irff.tools.reax_debug import grada,debug_plot,get_v
from irff.tools.reax_debug import compare_u,torsion,allgrad,gradb,gradt,get_spv
from irff.tools.reax_plot  import plot_delta,plbo,plf4
from irff.tools.reax_gulp import debug_h,debug_pen,debug_bo,debug_be
import argh
import argparse

dirs={'nm7_0':'nm7-0.traj',
      'nm7_1':'nm7-1.traj',
      'nm7_2':'nm7-2.traj',
      'nm7_3':'nm7-3.traj',
      'nm7_4':'nm7-4.traj',
      'nm10_0':'nm10-0.traj',
      'nm10_1':'nm10-1.traj',
      'nm10_2':'nm10-2.traj',
      'nm10_3':'nm10-3.traj',
      'case0':'case-0.traj',
      'n2co1':'/home/feng/siesta/n2co1'
      }

nbatchs={'nm1':1,'nm10':1} # 'hb4':2,'hb6_1':2,'oo':2
direcs = {}
for mol in dirs:
    nb = nbatchs[mol] if mol in nbatchs else 1
    for i in range(nb):
        direcs[mol+'-'+str(i)] = dirs[mol]
batch = 50

# system('rm *.pkl')
def gb(direcs=direcs):
    ''' variables: bo1_C-H '''
    gradb(direcs=direcs,
          v='rosi',bd='C-H',
          batch=batch) 

def ga():
    ''' variables: bo1_C-H '''
    grada(direcs=direcs,batch=batch) 

def gt():
    ''' variables: bo1_C-H '''
    gradt(direcs=direcs,v='rosi_C-H',batch=batch) 

def ag():
    allgrad(direcs=direcs,batch=50)

def spv():
    ''' variables: bo1_C-H '''
    get_spv(direcs=direcs,batch=batch) 

def v():
    ''' variables: bo1_C-H '''
    get_v(direcs=direcs,batch=batch) 

def dp():
    debug_pen()

def dh():
    debug_h(direcs={'ch4':'/home/feng/siesta/ch4'},
            gulp_cmd='/home/feng/gulp/Src/gulp<inp-gulp >gulp.out')

def dbe():
    debug_be(direcs={'ch4':'/home/feng/siesta/ch4'})

def db():
    debug_bo()

def pd():
    plot_delta(direcs=direcs)

def pb():
    plbo(lab='sigma')

def pf():
    plf4()

def t():
    torsion(direcs=direcs)


if __name__ == '__main__':
   ''' use commond like ./bpd.py t to run it
       d: debug
       t: test
       c: compare between GULP and BPNN
       g: get gradient of all vairables
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [ga,gb,gt,v,dp,db,dbe,dh,t,ag,pd,pb,pf,spv])
   argh.dispatch(parser)


