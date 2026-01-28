#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pyvista as pv
from ase.io import read
from irff.md.gulp import get_gulp_forces
from irff.irff import IRFF
from irff.molecule import press_mol

# ------------------- Forces from GULP --------------------
atoms  = read('md_eps.traj',index=-1)
atoms  = press_mol(atoms)
atoms_ = read('POSCAR.compare',index=-1)
atom_name = atoms.get_chemical_symbols()
cell   = atoms.get_cell()

# ----------------- Forces from autograd ------------------
#atoms.calc = IRFF(atoms=atoms, libfile='ffield.json',nn=True)
#forces = atoms.get_forces()
natom  = len(atoms)
points = atoms.positions
points_= atoms_.positions
# point_cloud = pv.PolyData(points)

## get image points according PBC conditions -------------
rcut   =  1.7
# points = np.array(points)

#------------------------ 计算成键 -------------------------
bds  =  [ ]
bds_ =  [ ]
for i in range(natom-1):
    for j in range(i+1,natom):
        vr_ = points[j] - points[i]
        r_  = np.sqrt(np.sum(vr_*vr_))
        if i < natom and j < natom:
           if atom_name[i]=='H' and  atom_name[j]=='H':
              continue
        if r_<rcut:
           bds.append([2,i,j]) 

for i_ in range(natom,2*natom-1):
    for j_ in range(i_+1,2*natom):
        i = i_ - natom
        j = j_ - natom
        vr_ = points_[j] - points_[i]
        r_  = np.sqrt(np.sum(vr_*vr_))
        if atom_name[i]=='H' and  atom_name[j]=='H':
           continue
        if r_<rcut:
           bds_.append([2,i,j])

#------------------------ 设置ploter ----------------------
#pv.global_theme.colorbar_horizontal.width = 0.2
pv.global_theme.transparent_background = True
p = pv.Plotter(off_screen=False,window_size=(2400,1800))
p.set_background('white')
#p.set_scale(xscale=4, yscale=4, zscale=4, reset_camera=True)
p.show_axes()
#-------------------  定义原子颜色和大小　--------------------
radius = {'C':0.40,'H':0.16,'N':0.33,'O':0.30}
colors = {'C':'black','H':'white','N':'deepskyblue','O':'m'}
opac   = 1 
#----------------------- 画出原子键　-----------------------
bonds        = pv.PolyData()
bonds.points = points

bonds.lines  = bds
tube         = bonds.tube(radius=0.12)
p.add_mesh(tube,pbr=True,metallic=3/4, color='y',roughness=2/5, opacity=1.0,smooth_shading=True)

if bds_:
   bonds = pv.PolyData()
   bonds.points = points_

   bonds.lines = bds_
   tube = bonds.tube(radius=0.1)
   p.add_mesh(tube,pbr=True,metallic=3/4, color='r',roughness=2/5, opacity=opac,smooth_shading=True)

#------------------------ 画出晶胞　------------------------
vertices = np.array([[0, 0, 0], cell[0], cell[0] + cell[1], cell[1],           
                        cell[1]+cell[2], cell[0]+cell[1]+cell[2], cell[0]+cell[2],
                        cell[2],cell[1]+cell[2],cell[1],[0,0,0],cell[2],cell[2]+cell[1],cell[1],
                        cell[1]+cell[0],cell[0],cell[2]+cell[0],cell[2]+cell[0]+cell[1],
                        cell[0]+cell[1],cell[0]])  # 单晶胞的所有顶点，共八个
box = pv.lines_from_points(vertices,close=False)
p.add_mesh(box,line_width=8,color='dodgerblue',metallic=1/8)

#p.view_vector((0,-1 , 0), (0, 1,0))
#p.view_vector((0,0 , -1), (0, 0,1))
p.camera_position = 'xy'
p.camera.zoom(2.0) 

# img_again = p.screenshot()
p.save_graphic('{:s}'.format('compare.svg'))
p.show(auto_close=False)
# p.screenshot(transparent_background=True,filename='{:s}'.format(args.geo.split('.')[0]+'.png'))
p.close()

