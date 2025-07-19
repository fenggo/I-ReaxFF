#!/usr/bin/env python
import sys
import argparse
from os import  system,mkdir
from os.path import exists
import json as js
import numpy as np
import time
import torch
from ase import Atoms
from irff.reaxff_torch import ReaxFF_nn
from irff.data.ColData import ColData
from irff.reax_force_data import reax_force_data
from irff.intCheck import init_bonds,check_tors

getdata = ColData()

dataset = {}
data    = {}
strucs  = ['ch4w2']

# weight_e = {'others':0.010,'tnt':0.005,'fox':0.005,'hmx':0.001,'tkx3':0.001,
#             'cl20c':0.01,'cf11':0.01}
# weight_f = {'tkx2':5.0,'c3h8':1,'tnt':0.5,'fox':0.1,'hmx':0.04,'tkx3':0.1,
#             'cl20c':0.1,'cf11':0.01}

batch        = 70
# batchs       = {'tkx2':batch,'fox7':150,'fox':500,'tnt':500,
#                 'cf11':172,
#                 'others':300}
batchs       = {'ch4w2':200,
                'others':300}
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

constants  = ['valboc','lp2_H',#'lp1',#'lp2',
              'theta0','val8',
              'valang','val9','val10','vale',
              'cot1','cot2','coa1','coa2','coa3','coa4',
              'pen1','pen2','pen3','pen4',
              #'vdw1','gammaw','rvdw','alfa','Devdw',
              'ovun5', 'ovun6','ovun7','ovun8',
              #'ovun2','ovun3','ovun4','ovun1',
              'val','lp3',
              'hb1','hb2',
              'acut','cutoff']    
#  'val7':(0.0,10.0),
#  'lp2':(0.0,999.0),'pen1':(-60.0,196.0),'ovun1':(0.0,2.0)
 
clip = {'Devdw':[0,0.856],
        'lp2':[0.0,40.0],'lp2_H':[0.0,0.0],
        #'pen1':[-45,87.848], 
        'ovun1':[0.0,1.0],#'ovun1_H-H':[0.0,0.9039],
        'Dehb':[-3.998,0.0],'Dehb_C-H-O':[-3.9,-0.35],
        'rohb':[1.877,2.392],'hb1':[2.72,3.64],'hb2':[18.7,19.64],
        'rvdw':[1.755,2.46],'rvdw_C':[1.864,2.399],'rvdw_O':[1.84,2.50],'rvdw_C-N':[1.7556,2.399],
        'val1':[0.0,70], 'val2':[0.05,6.55],
        'val3':[10,36],'val4':[0.0,3.0],'val5':[0.1,1.5],# cause NaN !!
        #'coa1':[-0.16,0.0],
        'tor1':[-8.26,-0.001],'tor2':[0.41,8.16],'tor3':[0.041,5.0],'tor4':[0.05,1.0],
        'V2':[0,89.5], # 'acut':[0,0.002798],
        'vdw1':[0.48,8.0] }

# while True:          # clip values
#     for key in clip:
#         if isinstance(clip[key],list):
#            clip[key][0] = clip[key][0]*0.99
#            clip[key][1] = clip[key][1]*0.99

rn = ReaxFF_nn(dataset=dataset,data=data,
            weight_energy=weight_e,
            weight_force=weight_f,
            cons=constants,
            libfile='ffield.json',
            clip=clip,
            screen=True,
            lambda_bd=1000.0,
            lambda_pi=0.0,
            lambda_reg=0.0001,
            lambda_ang=0.0,
            device={'cf21-1':'cpu:0','others':'cuda'})
data = rn.data
# print(rn.cons)
parser = argparse.ArgumentParser(description='./train_torch.py --e=1000')
parser.add_argument('--e',default=100001 ,type=int, help='the number of epoch of train')
parser.add_argument('--l',default=0.0001 ,type=float, help='learning rate')
args = parser.parse_args(sys.argv[1:])

optimizer = torch.optim.AdamW(rn.parameters(), lr=args.l)

if not exists('ffields'):
   mkdir('ffields')

n_epoch = 10001   
for epoch in range(n_epoch):
    los   = []
    los_e = []
    los_f = []
    los_p = []
    start = time.time()
    optimizer.zero_grad()
    for st in rn.strcs:
        E,F  = rn(st)             # forward
        loss = rn.get_loss(st)
        loss.backward(retain_graph=False)
        los.append(loss.item())
        los_e.append(rn.loss_e.item()/rn.natom[st])
        los_f.append(rn.loss_f.item()/(rn.natom[st]))
        los_p.append(rn.loss_penalty.item())
    optimizer.step()        # update parameters 
    
    if rn.ic.clip['V2'][1]>0:
       rn.ic.clip['V2'][1] -= 0.0001
    # if rn.ic.clip['V1'][1]>3:
    #    rn.ic.clip['V1'][1] -= 0.0001
    # if rn.ic.clip['tor2'][1]>0.3:
    #    rn.ic.clip['tor2'][1] -= 0.002
    rn.clamp()              # contrain the paramters
    los_ = np.mean(los)

    use_time = time.time() - start
    print( "eproch: {:5d} loss : {:10.5f} energy: {:7.5f} force: {:7.5f} pen: {:10.5f} time: {:6.3f}".format(epoch,
            los_,np.mean(los_e),np.mean(los_f),np.mean(los_p),use_time))
    if np.isnan(los_):
       break
    if epoch%100==0:
       rn.save_ffield('ffields/ffield_{:d}.json'.format(epoch))
       # rn.save_ffield('ffield.json')
       print('\n-------------------- Loss for Batches -------------------')
       print('-   Batch     TolLoss     EnergyLoss    ForceLoss       -')
       print('---------------------------------------------------------')
       for st,l,l_e,l_f,l_p in zip(rn.strcs,los,los_e,los_f,los_p):
           print('  {:10s} {:11.5f} {:10.5f}   {:10.5f}'.format(st,l,l_e,l_f))
       print('---------------------------------------------------------\n')
       if epoch==0:
          for st in rn.strcs:
              try:
                  estruc = np.mean(rn.dft_energy[st].detach().numpy()  - rn.E[st].detach().numpy())
              except RuntimeError:
                  estruc = np.mean(rn.dft_energy[st].cpu().detach().numpy()  - rn.E[st].cpu().detach().numpy())
              print('Estruc  {:10s}:   {:10.5f}'.format(st,estruc))
          print('---------------------------------------------------------\n')
rn.close()
