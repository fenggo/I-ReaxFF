#!/usr/bin/env python3
# coding: utf-8
import sys
import numpy as np
import argparse
from ase.optimize import BFGS,QuasiNewton
from ase.constraints import StrainFilter,FixAtoms
from ase.vibrations import Vibrations
from ase.io import read,write
from irff.irff import IRFF
from irff.md.irmd import IRMD
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.md.verlet import VelocityVerlet
from ase import units
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

help_ = './md.py --s=100 --g=POSCAR'

parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--gen',default='md.traj', help='atomic configuration')
parser.add_argument('--index',default=-1,type=int, help='the index in atomic configurations')
parser.add_argument('--step',default=100,type=int, help='the step of MD simulations')
parser.add_argument('--T',default=300,type=int ,help='the Temperature of MD simulations')
parser.add_argument('--r',default=0,type=int ,help='if run relax mode, i.e. relax the structure')
parser.add_argument('--b',default=0.96,type=float,help='the relax parameter')
parser.add_argument('--f',default=[],type=list,help='free atoms')
parser.add_argument('--m',default=1,type=int,help='manybody interaction flag')
args = parser.parse_args(sys.argv[1:])


def moleculardynamics():
    # opt(gen=gen)
    atoms = read(args.gen,index=args.index)#*[2,2,1]
    # ao    = AtomDance(atoms,bondTole=1.35)
    # atoms = ao.bond_momenta_bigest(atoms)
    
    f_= args.f if args.f else None
    nomb = False if args.m else True 
    
    irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=args.step,gen=args.gen,Tmax=10000,
                 freeatoms=f_,beta=args.b,
                 ro=0.8,rmin=0.5,initT=args.T,
                 ffield='ffield.json',
                 nomb=nomb,nn=True)
    irmd.run()
    mdsteps= irmd.step
    Emd  = irmd.Epot
    irmd.close()

def train_gp():
    dataset = {}
    strucs = ['tkx','tkx2']
    # strucs = ['tkxmd']

    trajdata = ColData()
    for mol in strucs:
        trajs = trajdata(label=mol,batch=50)
        dataset.update(trajs)

    bonds = ['C-C','C-H','C-N','H-O','C-O','H-H','H-N','N-N','O-N','O-O'] 
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = {}
    for bd in bonds:
        # for i,bp in enumerate(Bp[bd]):
        #     print(i,R[bd][i],D[bd][i][1],np.sum(B[bd][i]),E[bd][i])
        if bd not in D:
           continue
        D_  = np.array(D[bd])
        B_  = np.array(B[bd])

        print('Gaussian Process for {:s} bond ...'.format(bd))
        gp[bd] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9,
                                        optimizer=None) # fmin_l_bfgs_b
        gp[bd].fit(D_, B_)
        # print(gp[bd].kernel_)
        print('the score of exsiting data: ',gp[bd].score(D_, B_))
        D_md,Bp_md,B_md,R_md,E_md = get_md_data(images=None, traj='md.traj', bonds=['O-N'],ffield='ffieldData.json')
        
        if bd not in D_md:
           continue
        D_  = np.array(D_md[bd])
        B_  = np.array(B_md[bd])
        print('the score of new data: ',gp[bd].score(D_, B_))
        
        B_pred, std_pred = gp[bd].predict(D_, return_std=True)
        # print(len(D[bd]))
        # print(len(D_md[bd]))
        D[bd].extend(D_md[bd])
        B[bd].extend(B_pred.tolist())
        Bp[bd].extend(Bp_md[bd])
 
    train(Bp,D,B,E,bonds=bonds,step=step,fitobj='BO',learning_rate=0.0001)
    
def md_gp(timestep=0.1,print_interval=1,totstep=10000):
    atoms = read(args.gen,index=args.index)#*[2,2,1]
    atoms.calc= IRFF(atoms=atoms,libfile='ffield.json',nn=True)
    atoms.calc.get_bond_energy(atoms=atoms)

    gp = train_gp()

    def printenergy(atoms=atoms,gp=gp):
        ''' print out information '''

    dyn = VelocityVerlet(atoms, timestep*units.fs,trajectory='md.traj') 
    dyn.attach(printenergy,interval=print_interval)
    dyn.run(totstep)


if __name__ == '__main__':
   ''' run this script as :
        ./md.py --s=100 --g=md.traj 
        s 模拟步长
        g 初始结构'''
   # moleculardynamics()
   md_gp()
