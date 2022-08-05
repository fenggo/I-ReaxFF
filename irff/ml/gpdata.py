''' data preparation for a GaussianProcessRegressor
'''
import sys
import argh
import argparse
from ase.io.trajectory import Trajectory
from irff.irff_np import IRFF_NP
# dataset = {'dia-0':'data/dia-0.traj',
#            # 'dia-1':'data/dia-1.traj',
#            # 'dia-2':'data/dia-2.traj',
#            # 'dia-3':'data/dia-3.traj',
#            'gp2-0':'data/gp2-0.traj',
#            'gp2-1':'data/gp2-1.traj',
#            }

# getdata = ColData()
# strucs = ['h22',
#           'ch2',
#           'cn2',
#           'c6',
#           ]
# #strucs = ['c2h4','c2h6',]
# batchs = {'others':50}

# for mol in strucs:
#     b = batchs[mol] if mol in batchs else batchs['others']
#     trajs = getdata(label=mol,batch=b)
#     dataset.update(trajs)

# from sklearn.datasets import make_friedman2
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
# kernel = DotProduct() + WhiteKernel()
# gpr = GaussianProcessRegressor(kernel=kernel,
# ...         random_state=0).fit(X, y)
# gpr.score(X, y)
# 0.3680...
# gpr.predict(X[:2,:], return_std=True)


def get_data(dataset={'nm-0': 'nm-0.traj'}, bonds=['C-C'],ffield='ffield.json'):
    D, Y = {}, {}
    R, Bp,B = {}, {}, {}

    for bd in bonds:
        D[bd] = []
        Y[bd] = []
        Bp[bd] = []
        B[bd] = []
        R[bd] = []
        
    for traj in dataset:
        images = Trajectory(dataset[traj])

        ir = IRFF_NP(atoms=images[0],
                     libfile=ffield,
                     nn=True, vdwnn=False)

        for i, atoms in enumerate(images):
            energy = atoms.get_potential_energy()
            ir.calculate(atoms)

            for ii in range(ir.natom-1):
                for jj in range(ii+1, ir.natom):
                    if ir.bop[ii][jj] > 0.0001:
                        bd = ir.atom_name[ii] + '-' + ir.atom_name[jj]
                        bdr = ir.atom_name[jj] + '-' + ir.atom_name[ii]

                        if bd in bonds:
                            D[bd].append([ir.Deltap[ii]-ir.bop[ii][jj], ir.bop[ii][jj], ir.Deltap[jj]-ir.bop[ii][jj]])
                            Bp[bd].append([ir.bop_si[ii][jj], ir.bop_pi[ii][jj], ir.bop_pp[ii][jj]])
                            B[bd].append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])
                            R[bd].append(ir.r[ii][jj])
                            Y[bd].append(ir.esi[ii][jj])
                        elif bdr in bonds:
                            D[bdr].append([ir.Deltap[jj]-ir.bop[ii][jj], ir.bop[ii][jj], ir.Deltap[ii]-ir.bop[ii][jj]])
                            Bp[bdr].append([ir.bop_si[ii][jj], ir.bop_pi[ii][jj], ir.bop_pp[ii][jj]])
                            B[bdr].append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])
                            R[bdr].append(ir.r[ii][jj])
                            Y[bdr].append(ir.esi[ii][jj])
        del ir
    return D, Bp,B, R, Y


def get_md_data(images=None, traj='md.traj', bonds=['C-C']):
    if images is None:
        images = Trajectory(traj)
    # mol_   = traj.split('.')[0]
    # mol    = mol_.split('-')[0]
    D, Y = {}, {}
    R, Bp,B = {}, {}, {}
    for bd in bonds:
        D[bd] = []
        Y[bd] = []
        Bp[bd] = []
        B[bd] = []
        R[bd] = []

    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nn=True, vdwnn=False)

    for i, atoms in enumerate(images):
        ir.calculate(atoms)

        for ii in range(ir.natom-1):
            for jj in range(ii+1, ir.natom):
                if ir.bop[ii][jj] > 0.0001:
                    bd = ir.atom_name[ii] + '-' + ir.atom_name[jj]
                    bdr = ir.atom_name[jj] + '-' + ir.atom_name[ii]
                    if bd in bonds:
                        D[bd].append([ir.Deltap[ii]-ir.bop[ii][jj], ir.bop[ii][jj], ir.Deltap[jj]-ir.bop[ii][jj]])
                        Bp[bd].append([ir.bop_si[ii][jj],ir.bop_pi[ii][jj],ir.bop_pp[ii][jj]])
                        B[bd].append([ir.bosi[ii][jj],ir.bopi[ii][jj],ir.bopp[ii][jj]])
                        R[bd].append(ir.r[ii][jj])
                        Y[bd].append(ir.esi[ii][jj])
                    elif bdr in bonds:
                        D[bdr].append([ir.Deltap[jj]-ir.bop[ii][jj], ir.bop[ii][jj], ir.Deltap[ii]-ir.bop[ii][jj]])
                        Bp[bdr].append([ir.bop_si[ii][jj], ir.bop_pi[ii][jj], ir.bop_pp[ii][jj]])
                        B[bdr].append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])
                        R[bdr].append(ir.r[ii][jj])
                        Y[bdr].append(ir.esi[ii][jj])
    del ir
    return D, Bp, B, R, Y

def get_bond_data(ii, jj, images=None, traj='md.traj', bonds=None):
    if images is None:
        images = Trajectory(traj)
    D = []
    Y = []
    Bp= []
    B = []
    R = []
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nn=True, vdwnn=False)
                 
    if bonds is None:
        bonds = [ir.atom_name[ii]+'-'+ir.atom_name[jj]]

    bd = ir.atom_name[ii] + '-' + ir.atom_name[jj]
    bdr = ir.atom_name[jj] + '-' + ir.atom_name[ii]

    for i, atoms in enumerate(images):
        ir.calculate(atoms)
        if bd in bonds:
            D.append([ir.Deltap[ii]-ir.bop[ii][jj], ir.bop[ii][jj], ir.Deltap[jj]-ir.bop[ii][jj]])
        elif bdr in bonds:
            D.append([ir.Deltap[jj]-ir.bop[ii][jj],ir.bop[ii][jj],ir.Deltap[ii]-ir.bop[ii][jj]])
        Bp.append([ir.bop_si[ii][jj],ir.bop_pi[ii][jj],ir.bop_pp[ii][jj]])
        B.append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])                   
        R.append(ir.r[ii][jj])
        Y.append(ir.esi[ii][jj])
    del ir
    return D, Bp, B, R, Y

