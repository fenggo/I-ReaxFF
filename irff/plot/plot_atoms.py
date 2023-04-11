#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pyvista as pv
from ase.io import read
from irff.md.gulp import get_gulp_forces
from irff.irff import IRFF

parser = argparse.ArgumentParser(description='write the uspex style Z-matrix infomation')
parser.add_argument('--geo',default='POSCAR',type=str, help='geomentry file name')
parser.add_argument('--i',default=0,type=int, help='the i_th frame in traj file')
args = parser.parse_args(sys.argv[1:])

# ------------------- Forces from GULP --------------------
atoms  = read('gulp.traj',index=-1)
atoms  = get_gulp_forces([atoms]) 
# ----------------- Forces from autograd ------------------
#atoms.calc = IRFF(atoms=atoms, libfile='ffield.json',nn=True)
#forces = atoms.get_forces()

natom  = len(atoms)
points = atoms.positions
point_cloud = pv.PolyData(points)
point_cloud['vectors'] = atoms.get_forces()
arrows = point_cloud.glyph(orient='vectors',scale=False,factor=2.0)

## get image points according PBC conditions -------------

xi     = np.expand_dims(points,axis=0)
xj     = np.expand_dims(points,axis=1)
vr     = xj-xi

cell   = atoms.get_cell()
rcell  = np.linalg.inv(cell)

xf     = np.dot(points,rcell)
vrf_   = np.dot(vr,rcell)
vrf    = np.where(vrf_-0.5>0,vrf_-1.0,vrf_)
vrf    = np.where(vrf+0.5<0,vrf+1.0,vrf)  
vr     = np.dot(vrf,cell)
r      = np.sqrt(np.sum(vr*vr,axis=2))

rcut   =  1.7
points = list(points)

for i in range(natom-1):               # 增加与影像原子成键的端点
    for j in range(i+1,natom):
        if r[i][j]<rcut:
           vr_ = points[j] - points[i]
           r_  = np.sqrt(np.sum(vr_*vr_))
           if r_>rcut:
              points.append(points[i] + 0.5*vr[j][i])
              points.append(points[j] + 0.5*vr[i][j])
points = np.array(points)
natom_ = len(points)
#------------------------ 计算成键 -------------------------
bds =  [ ]
for i in range(natom_-1):
    for j in range(i+1,natom_):
        if i>=natom and j>=natom:
           continue
        vr_ = points[j] - points[i]
        r_  = np.sqrt(np.sum(vr_*vr_))
        if r_<rcut:
           bds.append([2,i,j]) 

#------------------------ 设置ploter ----------------------
pv.global_theme.colorbar_horizontal.width = 0.2
p = pv.Plotter(off_screen=True)
p.set_background('white')
p.set_scale(xscale=4, yscale=4, zscale=4, reset_camera=True)
p.show_axes()
#------------------------ 画出原子　------------------------
for atom in atoms:
    sphere = pv.Sphere(radius=0.3, center=(atom.x,atom.y,atom.z))
    p.add_mesh(sphere, color='grey', pbr=True, metallic=2/4, roughness=2/5)

#----------------------- 画出原子键　-----------------------
bonds = pv.PolyData()
bonds.points = points

bonds.lines = bds
tube = bonds.tube(radius=0.15)
# tube.plot(smooth_shading=True,pbr=True, metallic=2/4,)
p.add_mesh(tube,pbr=True,metallic=3/4, roughness=2/5, smooth_shading=True)

#------------------------ 画出力矢量　----------------------
p.add_mesh(arrows, color='red',pbr=True, smooth_shading=True)

#------------------------ 画出晶胞　------------------------
vertices = np.array([[0, 0, 0], cell[0], cell[0] + cell[1], cell[1], 
                      cell[1]+cell[2], cell[0]+cell[1]+cell[2], cell[0]+cell[2],
                      cell[2],cell[1]+cell[2],cell[1],[0,0,0],cell[2],cell[2]+cell[1],cell[1],
                      cell[1]+cell[0],cell[0],cell[2]+cell[0],cell[2]+cell[0]+cell[1],
                      cell[0]+cell[1],cell[0]])
box = pv.lines_from_points(vertices,close=True)
p.add_mesh(box,line_width=2,color='blue')

p.view_vector((0,-1 , 0), (0, 1,0))
#p.camera_position = 'xy'
p.save_graphic('ball.svg',raster=False)
#p.show(auto_close=False)
#p.screenshot()
p.close()


