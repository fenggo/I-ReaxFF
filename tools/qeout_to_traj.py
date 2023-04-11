#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.visualize import view
from ase.units import create_units
from ase.dft.kpoints import kpoint_convert
from ase.calculators.singlepoint import SinglePointDFTCalculator,SinglePointKPoint
from ase.atoms import Atoms
from irff.dft.qe import read_espresso_out
import numpy as np
import os


###获取所有out文件的文件名##########################################
def getSuffixFileName(path,suffix):
    # 获取指定目录下的所有指定后缀的文件名
    SuffixFileName=[]
    f_list = os.listdir(path)#返回文件名
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] ==suffix:
            SuffixFileName.append(i)
            #print(i)
    return SuffixFileName

path=os.getcwd()
suffix='.out'

SuffixFileName=getSuffixFileName(path,suffix)
print(SuffixFileName)

###定义格式转换函数###############################################
def trans_Qeout_to_traj(filename):
    file_title=filename.split('.')
    output_filename=file_title[0]+'.traj'

    his = TrajectoryWriter(output_filename,mode='w')
    images  = read_espresso_out(filename)
    #images_ = []
    for image in images:
        his.write(atoms=image)
        #images_.append(image)

    his.close()
    #view(images_)

####同时对所有out文件进行格式转换#######################################
for FileName in SuffixFileName:
    trans_Qeout_to_traj(FileName)
