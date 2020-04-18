#!/usr/bin/env python
from __future__ import print_function
from os import system, getcwd, chdir,listdir,environ
from os.path import isfile,exists,isdir
from irff.irnn_debug import grada,compare_bo,debug_plot,compare_d,compare_eb,get_v
from irff.irnn_debug import compare_u,allgrad,gradb,compare_f,get_f4
from irff.irnn_plot  import plot_delta,plbo,plf4
from irff.reax_gulp import debug_hb,debug_pen,debug_bo,debug_be,debug_ang,debug_eu
import argh
import argparse


direcs={'hb':'/home/feng/siesta/train2/hb',
      'hb-1':'/home/feng/siesta/train2/hb1',
      'hb-2':'/home/feng/siesta/train2/hb2',
      'hb-3':'/home/feng/siesta/train2/hb3',
      'hb-4':'/home/feng/siesta/train2/hb4',
      'hb-5':'/home/feng/siesta/train2/hb5',
      'hb-6':'/home/feng/siesta/train2/hb6'}

# system('rm *.pkl')
def gb(direcs={'ch4':'/home/feng/siesta/ch4'}):
    ''' variables: bo1_C-H '''
    gradb(direcs=direcs) 

def ga():
    ''' variables: bo1_C-H '''
    grada(direcs={'ch4':'/home/feng/siesta/ch4'}) 

def dp():
    debug_pen()

def dh():
    debug_hb()

def da():
    debug_ang(direcs={'cho-4':'/home/feng/siesta/train2/cho4/'},
              gulp_cmd='/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out')

def du():
    debug_eu(direcs={'cho-4':'/home/feng/siesta/train2/cho4/'},
              gulp_cmd='/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out')

def dbe():
    debug_be(direcs={'ch4':'/home/feng/siesta/ch4'})

def db():
    debug_bo()

def pd():
    plot_delta(direcs={'ch4':'/home/feng/siesta/ch4'})

def pb():
    plbo(lab='sigma')

def pf():
    plf4()

def bo():
    compare_bo('/home/gfeng/cpmd/train/nmr/dop')

def v():
    get_v()

def cd():
    compare_d('/home/gfeng/cpmd/train/nmr/nm13')

def eb():
    compare_eb('/home/gfeng/cpmd/train/nmr/dop')

def u():
    compare_u('/home/gfeng/cpmd/train/nmr/nm13')

def t():
    compare_tor('/home/gfeng/cpmd/train/nmr/nm13')

def ag():
    allgrad()

def f4():
    get_f4(direcs={'hb-6':'/home/feng/siesta/train2/hb6'})




if __name__ == '__main__':
   ''' use commond like ./bpd.py t to run it
       d: debug
       t: test
       c: compare between GULP and BPNN
       g: get gradient of all vairables
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [ga,gb,v,dp,db,da,du,dbe,dh,cd,bo,eb,u,t,ag,f4,pd,pb,pf])
   argh.dispatch(parser)


