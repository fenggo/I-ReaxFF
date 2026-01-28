#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pyvista as pv
from ase.io import read
from irff.md.gulp import get_gulp_forces
from irff.molecule import Molecules,moltoatoms

help_ = './plotatoms.py --g=graphene.gen'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--geo',default='gulp.traj',type=str, help='geomentry file name')
parser.add_argument('--i',default=0,type=int, help='the i_th frame in traj file')
parser.add_argument('--f',default=0,type=int, help='whether plot the forces')
# parser.add_argument('--o',default=0,type=int, help='plot atoms out of the box with opacity')  
parser.add_argument('--camera_position',default='xy',type=str, help='whether plot the box') 
args = parser.parse_args(sys.argv[1:])

#-------------------  定义原子颜色和大小　--------------------
radius = {'C':0.36,'H':0.16,'N':0.30,'O':0.28}
colors = {'C':'black','H':'white','N':'deepskyblue','O':'m'}

def parse_atoms(atoms):
    natom  = len(atoms)
    arrows = None

    points = atoms.positions
    point_cloud = pv.PolyData(points)

    ## get image points according PBC conditions -------------
    xi     = np.expand_dims(points,axis=0)
    xj     = np.expand_dims(points,axis=1)
    vr     = xj-xi
    r      = np.sqrt(np.sum(vr*vr,axis=2))

    rcut   =  1.7
    points = list(points)
    points = np.array(points)
    natom = len(points)
    #------------------------ 计算成键 -------------------------
    bds  =  [ ]
    for i in range(natom-1):
        for j in range(i+1,natom):                
            vr_ = points[j] - points[i]
            r_  = np.sqrt(np.sum(vr_*vr_))
            # print(r_) 
            # if atom_name[i]=='H' and  atom_name[j]=='H':
            #       continue
            if r_<rcut:
               bds.append([2,i,j]) 
    return points,bds,arrows

#------------------------ 设置ploter ----------------------
#pv.global_theme.colorbar_horizontal.width = 0.2
pv.global_theme.transparent_background = True
p = pv.Plotter(off_screen=False,window_size=(2400,1800))
p.set_background('white')
#p.set_scale(xscale=4, yscale=4, zscale=4, reset_camera=True)
p.show_axes()

atoms  = read(args.geo,index=args.i)
m_     = Molecules(atoms,rcut={'O-H':1.2,'others': 1.6},check=True)
print(m_)

for m in m_:
    atoms_ = moltoatoms([m])
    points,bds,arrows = parse_atoms(atoms_)
    natom = len(atoms_)

    #------------------------ 画出原子　------------------------
    opac =  1 
    if natom==21:
       c = "#10D745"  
    else:
       c = "#EC062C"
   
    for i,atom in enumerate(atoms_):
        sphere = pv.Sphere(radius=radius[atom.symbol], center=(atom.x,atom.y,atom.z))
        # c = colors[atom.symbol] 
        p.add_mesh(sphere, color=c, pbr=True, opacity=1.0,metallic=1/8, roughness=1/5)  

    #----------------------- 画出原子键　-----------------------
    bonds = pv.PolyData()
    bonds.points = points

    # print(bds)
    bonds.lines = bds
    tube = bonds.tube(radius=0.10)
    # tube.plot(smooth_shading=True,pbr=True, metallic=2/4,)
    p.add_mesh(tube,pbr=True,metallic=3/4,color=c, roughness=2/5, opacity=1.0,smooth_shading=True)

 
# p.view_vector((0,1,0))
# p.view_vector((0,0 , -1), (0, 0,1))
# p.camera_position = [-1, 1, 0.5]

p.camera_position = args.camera_position
p.camera.zoom(1.6) 

# img_again = p.screenshot()
p.save_graphic('{:s}'.format(args.geo.split('.')[0]+'.svg'))
p.show(interactive=True, auto_close=False)
# p.screenshot(transparent_background=True,filename='{:s}'.format(args.geo.split('.')[0]+'.png'))
p.close()

