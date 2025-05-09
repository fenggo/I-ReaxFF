#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pyvista as pv
from ase.io import read
from irff.md.gulp import get_gulp_forces
from irff.irff import IRFF
from irff.molecule import press_mol

help_ = './plotatoms.py --g=graphene.gen'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--geo',default='gulp.traj',type=str, help='geomentry file name')
parser.add_argument('--i',default=0,type=int, help='the i_th frame in traj file')
parser.add_argument('--r',default=1,type=int, help='repeat structure')
# parser.add_argument('--y',default=1,type=int, help='repeat structure in y direction')
# parser.add_argument('--z',default=1,type=int, help='repeat structure in z direction')
parser.add_argument('--f',default=0,type=int, help='whether plot the forces')
parser.add_argument('--b',default=1,type=int, help='whether plot the box')  
parser.add_argument('--o',default=0,type=int, help='plot atoms out of the box with opacity')  
parser.add_argument('--camera_position',default='xy',type=str, help='whether plot the box') 
args = parser.parse_args(sys.argv[1:])

# ------------------- Forces from GULP --------------------
atoms  = read(args.geo,index=args.i)
atoms  = press_mol(atoms)
x = y = z =2
if args.r:
   atoms  = atoms*(x,y,z)
 
atom_name = atoms.get_chemical_symbols()
if args.f:
   atoms  = get_gulp_forces([atoms]) 
# ----------------- Forces from autograd ------------------
#atoms.calc = IRFF(atoms=atoms, libfile='ffield.json',nn=True)
#forces = atoms.get_forces()
natom  = len(atoms)
natom_u= int(natom/(x*y*z)) if args.r else natom
points = atoms.positions
point_cloud = pv.PolyData(points)
if args.f:
   point_cloud['vectors'] = atoms.get_forces()
   arrows = point_cloud.glyph(orient='vectors',scale=False,factor=1.65)

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
bds  =  [ ]
bds_ =  [ ]
for i in range(natom_-1):
    for j in range(i+1,natom_):
        if i>=natom and j>=natom:
           continue                   
        vr_ = points[j] - points[i]
        r_  = np.sqrt(np.sum(vr_*vr_))
        # print(r_)
        if i < natom and j < natom:
           if atom_name[i]=='H' and  atom_name[j]=='H':
              continue
        if r_<rcut:
           if i < natom_u and j < natom_u:
              bds.append([2,i,j]) 
           else:
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
# colors = {'C':'grey','H':'whitesmoke','N':'blue','O':'red'}
colors = {'C':'black','H':'white','N':'deepskyblue','O':'m'}
# colors = {'C':'black','H':'white','N':'blue','O':'red'}
#bond_radius = {'C':0.15,'H':0.05,'N':0.15,'O':0.15}
#------------------------ 画出原子　------------------------
opac = 0.2 if args.o else 1 

for i,atom in enumerate(atoms):
    sphere = pv.Sphere(radius=radius[atom.symbol], center=(atom.x,atom.y,atom.z))
    if i< natom_u:
       p.add_mesh(sphere, color=colors[atom.symbol], pbr=True, opacity=1.0,
               metallic=1/8, roughness=1/5)
    else:
       p.add_mesh(sphere, color=colors[atom.symbol], pbr=True, opacity=opac,
                  metallic=1/8, roughness=1/5)  

#----------------------- 画出原子键　-----------------------
bonds = pv.PolyData()
bonds.points = points

# print(bds)
bonds.lines = bds
tube = bonds.tube(radius=0.12)
# tube.plot(smooth_shading=True,pbr=True, metallic=2/4,)
p.add_mesh(tube,pbr=True,metallic=3/4, roughness=2/5, opacity=1.0,smooth_shading=True)

if bds_:
   bonds = pv.PolyData()
   bonds.points = points

   bonds.lines = bds_
   tube = bonds.tube(radius=0.1)
   # tube.plot(smooth_shading=True,pbr=True, metallic=2/4,)
   p.add_mesh(tube,pbr=True,metallic=3/4, roughness=2/5, opacity=opac,smooth_shading=True)

#------------------------ 画出力矢量　----------------------
if args.f:
   p.add_mesh(arrows, color='red',pbr=True, smooth_shading=True)

#------------------------ 画出晶胞　------------------------
if args.r:
   cell[0] = cell[0]/x
   cell[1] = cell[1]/y
   cell[2] = cell[2]/z
if args.b:
   vertices = np.array([[0, 0, 0], cell[0], cell[0] + cell[1], cell[1],           
                           cell[1]+cell[2], cell[0]+cell[1]+cell[2], cell[0]+cell[2],
                           cell[2],cell[1]+cell[2],cell[1],[0,0,0],cell[2],cell[2]+cell[1],cell[1],
                           cell[1]+cell[0],cell[0],cell[2]+cell[0],cell[2]+cell[0]+cell[1],
                           cell[0]+cell[1],cell[0]])  # 单晶胞的所有顶点，共八个
   box = pv.lines_from_points(vertices,close=False)
   p.add_mesh(box,line_width=8,color='dodgerblue',metallic=1/8)

# p.view_vector((0,1,0))
# p.view_vector((0,0 , -1), (0, 0,1))
# p.camera_position = [-1, 1, 0.5]

p.camera_position = args.camera_position
p.camera.zoom(1.6) 

# img_again = p.screenshot()
p.save_graphic('{:s}'.format(args.geo.split('.')[0]+'.svg'))
# p.show(interactive=True, auto_close=False)
p.screenshot(transparent_background=True,filename='{:s}'.format(args.geo.split('.')[0]+'.png'))
p.close()

