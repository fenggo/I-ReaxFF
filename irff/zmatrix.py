from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import numpy as np


def stretch_atom(v,r,posi):
    posj = posi + r*v
    return posj

def rotate_atom(positions,i,j,k,l,r=None,ang=None,tor=None):
    vij = positions[i] - positions[j] 
    vkj = positions[k] - positions[j]
    rkj = np.sqrt(np.sum(np.square(vkj)))
    rij = np.sqrt(np.sum(np.square(vij)))
    
    ux  = vkj/rkj
    if l==-1:
       uij = vij/rij
       rk  = np.dot(uij,ux)
       vy  = uij - rk*ux
       # print(i,j,k,l,vy,uij,rk,ux)
       uy  = vy/np.sqrt(np.sum(np.square(vy)))
    else:
       vkl = positions[k] - positions[l] 
       rkl = np.sqrt(np.sum(np.square(vkl)))
       ukl = vkl/rkl
       rk  = np.dot(ukl,ux)
       vy  = ukl - rk*ux
       uy  = vy/np.sqrt(np.sum(np.square(vy))) 

    a   = ang*3.141593/180.0
    ox  = r*np.cos(a)*ux
    ro  = r*np.sin(a)
    oy  = ro*uy
    p   = ox + oy
    positions[i] = positions[j] + p

    if l != -1:
       vij = p
       uz  = np.cross(ux,uy)
       o_  = positions[j] + ox
       a   = tor*3.141593/180.0
       p   = ro*np.cos(a)*uy + ro*np.sin(a)*uz
       positions[i] = o_ + p
    return positions


def zmat_to_cartation(zmat,first=[0.0,0.0,0.0],second=[1.0,0.0,0.0],third=[0.0,1.0,0.0]):
    ''' '''
    positions = []
    atom_id = [z[1] for z in zmat]

    vij = np.array(second)-np.array(first)
    rij = np.sqrt(np.sum(np.square(vij)))
    mov_x = vij/rij

    vij = np.array(third)-np.array(first)
    rij = np.sqrt(np.sum(np.square(vij)))
    mov_y = vij/rij

    for i,z in enumerate(zmat):
        atomi = zmat[i][1]
        atomj = zmat[i][2]
        atomk = zmat[i][3]
        atoml = zmat[i][4]
        r     = zmat[i][5]
        ang   = zmat[i][6]
        tor   = zmat[i][7]

        if   i==0:
           positions.append(first)
        elif i==1:
           pos_ = stretch_atom(mov_x,zmat[i][5],positions[-1])
           positions.append(pos_)
        elif i==2:
           if ang !=180.0:
              mov_ = mov_y
           else:
              mov_ = mov_x
           pos_ = stretch_atom(mov_,zmat[i][5],positions[-1])
           positions.append(pos_)
           if ang != 180.0:
              positions = rotate_atom(positions,atomi,atomj,atomk,atoml,r=r,ang=ang,tor=tor)
        else:
           if ang !=180.0:
              if mov_[0] == mov_x[0] and mov_[1] == mov_x[1] and mov_[2] == mov_x[2]:
                 mov_ = mov_y
           pos_ = stretch_atom(mov_,zmat[i][5],positions[-1])
           positions.append(pos_)
           
           if ang !=180.0:
              positions = rotate_atom(positions,atomi,atomj,atomk,atoml,r=r,ang=ang,tor=tor)
    return positions


def zmat_to_atoms(zmat,first=[0.0,0.0,0.0],second=[1.0,0.0,0.0],third=[0.0,1.0,0.0],resort=True):
    zmat_,atom_id = zmat_resort(zmat)
    positions = zmat_to_cartation(zmat_,first=first,second=second,third=third)
    positions = np.array(positions)
    cell      = [np.max(positions[:,0],axis=0)+5.0,
                 np.max(positions[:,1],axis=0)+5.0,
                 np.max(positions[:,2],axis=0)+5.0]
    species   = [z[0] for z in zmat]

    species_ = species.copy()
    positions_ = positions.copy()

    if resort:
       for i,i_ in enumerate(atom_id):
           species_[i_]   = species[i]
           positions_[i_] = positions[i]

    atoms = Atoms(species_,positions_,cell=cell,pbc=[True,True,True])
    return atoms


def zmat_resort(zmat):
    atom_id = [z[1] for z in zmat]
    zmat_   = zmat.copy()
    for i,z in enumerate(zmat):
        if i==0:
           zmat_[i][1] = 0
        else:
           for j in range(1,5):
               if zmat[i][j] != -1:
                  zmat_[i][j] = atom_id.index(zmat[i][j])
    return zmat_,atom_id

