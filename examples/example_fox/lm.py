#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from irff.LearningMachine import LearningMachine
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
from train import clip,cons,batch,dataset,weight,mf_layer

# check_emol(dataset)

lm = LearningMachine(config='cf13.gen',
                     dataset=dataset,ncpu=8,batch=batch,
                     maxiter=5,
                     step=10000,md_step=50,col_frame=10,col_min_interval=4,
                     lossCriteria=6.5,accCriteria=0.5,
                     CheckZmat=False,
                     lambda_reg=0.005,lambda_bd=1000.0,lambda_me=0.01,
                     learn_method=2,dft_step=20, 
                     EngTole=0.01,dEtole=0.1,dEstop=2.0,
                     EnergyFunction=1,MessageFunction=3,
                     mf_layer=mf_layer,be_layer=[9,1],
                     mf_univeral_nn=None,#['C'],
                     be_univeral_nn=None,#['C-C','C-O'],
                     weight=weight,
                     #pi_clip=pi_clip,
                     #be_clip=be_clip,
                     optword='nocoul',cons=cons,clip=clip, 
                     writelib=1000,convergence=0.000001,
                     trainer=2,
                     dft='siesta',xcf='GGA',xca='PBE',basistype='split')
lm.run()
lm.close()

#   Use commond like: nohup ./lm.py >py.log 2>&1 &
#   to run it
#
