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
batchs       = {'ch4w2':batch,
                'others':300}
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

constants  = ['valboc',
              'Depi','Depp','be1','be2',
              'lp2_H','lp1','lp2',
              'theta0','val8',#'val1','val2','val3','val4',
              'valang','val9','vale','val10',
              'val1_H-O-O',
              'val2_O-C-O','val2_O-C-N',
              # 'val6',
              'cot1','cot2','coa1','coa2','coa3','coa4',
              'pen1','pen2','pen3','pen4',
              # 'vdw1','gammaw','rvdw','alfa','Devdw',
              'ovun5', 'ovun6','ovun7','ovun8',
              'ovun2','ovun3','ovun4','ovun1',
              'tor3',# 'tor1_N-C-N-N',
              'val','lp3',
              'hb1','hb2',
              # 'acut',
              'cutoff']    
#  'val7':(0.0,10.0),
#  'lp2':(0.0,999.0),'pen1':(-60.0,196.0),'ovun1':(0.0,2.0)
 
clip = {'Devdw':[0,0.856],
        'lp1':[0.0,30.0],'lp2':[0.0,60.0],'lp2_H':[0.0,0.0],
        'pen1':[-3.77,4.46], 
        'ovun1':[0.0,0.149],#'ovun1_H-H':[0.0,0.9039],
        'ovun5':[0.0,8.33],
        'Dehb':[-3.998,0.0],'Dehb_C-H-O':[-3.9,-0.35],
        'rohb':[1.877,2.392],'hb1':[2.72,3.64],'hb2':[18.7,19.64],
        'rvdw':[1.755,2.45],'rvdw_C':[1.864,2.399],'rvdw_O':[1.84,2.50],'rvdw_C-N':[1.7556,2.399],
        'val1':[0.0,70],'val1_H-O-H':[0.01,70], 
        'val2':[0.0,2.],'val4':[0.0,3.0],
        'val3':[10,36],'val5':[0.1,1.5],# cause NaN !!
        'val6':[0.0,6],'val7':[0.0,2.0], 'val9':[0.15,1.0],'val10':[1.0,6.0],
        'tor1':[-8.26,-0.001],'tor2':[0.41,0.456],'tor3':[0.041,1.5],'tor4':[0.05,1.0],
        'V1':[-33,33],'V2':[0,77.15], 'acut':[0.0009,0.0019],
        'vdw1':[0.48,8.0] }
bo_clip = {'N-N':[(2.0,0,9,0,9,0.0,0.001)],
           'C-N':[(2.0,6.5,7.2,3.2,3.7,0.0,0.001)]}

parser = argparse.ArgumentParser(description='./train_torch.py --e=1000')
parser.add_argument('--e',default=5001 ,type=int, help='the number of epoch of train')
parser.add_argument('--c',default=0 ,type=int, help='circulation')
parser.add_argument('--l',default=0.0001 ,type=float, help='learning rate')
args = parser.parse_args(sys.argv[1:])
# optimizer = torch.optim.Adadelta(rn.parameters(), lr=0.00003)
# rn.cuda()
# rn.compile() 

if not exists('ffields'):
   mkdir('ffields')

def fit():
    rn = ReaxFF_nn(dataset=dataset,# data=data,
                weight_energy=weight_e,
                weight_force=weight_f,
                cons=constants,
                # opt=['be1','be2','Desi','Depi','Depp'],
                libfile='ffield.json',
                clip=clip,bo_clip=bo_clip,
                EnergyFunction=1,
                screen=False,
                lambda_bd=3000.0,
                lambda_pi=0.0,
                lambda_reg=0.0001,
                lambda_ang=0.0,
                device={'cf21-1':'cpu:0','others':'cuda'})
    data = rn.data
    # print(rn.cons)
    optimizer = torch.optim.Adam(rn.parameters(), lr=args.l)
    
    n_epoch = args.e + 1
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
        if args.e>:
           optimizer.step()        # update parameters 
        
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
                  except TypeError:
                      estruc = np.mean(rn.dft_energy[st].cpu().detach().numpy()  - rn.E[st].cpu().detach().numpy())
                  print('  Estruc  {:10s}:   {:10.5f}'.format(st,estruc))
                  if args.e<=1:
                     st_ = st.split('-')[0]
                     emol = rn.MolEnergy_[st_] + estruc
                     rn.estruc[st].data = torch.clamp(rn.estruc[st].data,min=emol,max=emol)
              print('---------------------------------------------------------\n')
    rn.save_ffield('ffield.json')
    rn.close()
    
if __name__ == '__main__':
   if args.e>1 and args.c:
      while True:
            fit()
   else:
      fit()
