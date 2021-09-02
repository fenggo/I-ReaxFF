#!/usr/bin/env python
# coding: utf-8
from os import system,listdir,getcwd,chdir
from os.path import isdir


root_dir = getcwd()
# outs   = listdir(root_dir)

def rm_save(direc=None):
    direc_ = direc + '/'
    # chdir(direc)
    # print('Now in direc: ',getcwd())
    if isdir('pwscf.save'):
       system('rm -r pwscf.save')
       print('Find pwscf.save in dir {:s}'.format(direc))

    # cdir = getcwd()
    outs = listdir(direc_)
    # print(outs)
    for d in outs:
        d_ = direc+'/'+d
        if isdir(d_):
           rm_save(direc=d_)

    chdir(direc_)
    # return direc_


rm_save(direc=root_dir)

