#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from irff.LearningMachine import LearningMachine
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
from train import clip,strucs,cons,weight,bo_clip,pi_clip

dataset = {'h22-v':'aimd_h22/h22-v.traj',
        #    'dia-0':'data/dia-0.traj',
        #    'gp2-0':'data/gp2-0.traj',
        #    'gp2-1':'data/gp2-1.traj',
           }

getdata = ColData()


batch  = 50
batchs = {'others':batch }

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)
check_emol(dataset)


lm = LearningMachine(config='hc11.gen',
                     dataset=dataset,ncpu=8,batch=batch,
                     maxiter=0,
                     step=20000,col_frame=8,col_min_interval=5,
                     T=2000,
                     lossCriteria=3.0,accCriteria=0.5,
                     CheckZmat=True,
                     pi_clip=pi_clip,
                     bo_clip=bo_clip,
                     lambda_reg=0.001,lambda_bd=100.0,lambda_me=0.001,
                     learn_method=2,dft_step=4,
                     # beta=0.9,FirstAtom=30,FreeAtoms=[19,18,30,31], 
                     EngTole=0.01,dEtole=0.1,dEstop=2.0,
                     EnergyFunction=1,MessageFunction=3,
                     mf_layer=[9,1],be_layer=[9,1],
                     mf_universal_nn=None,              # ['C'],
                     be_universal_nn=None,              # ['C-C','C-O'],
                     weight=weight,
                     optword='nocoul',cons=cons,clip=clip,       # nocoul-nolone-nounder-noover
                     writelib=1000,convergence=0.04,
                     dft='siesta',xcf='GGA',xca='PBE',basistype='split')
lm.run()
lm.close()

