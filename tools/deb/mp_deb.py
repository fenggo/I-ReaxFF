#!/usr/bin/env python
import sys
import argparse
import numpy as np
from irff.ml.train import train
from irff.data.ColData import ColData
from irff.ml.fluctuation import morse
from irff.reax_nn import ReaxFF_nn

getdata = ColData()
dataset = {}

strucs = ['al64',
          'AlO', 
          'o22' ]

getdata = ColData()
for mol in strucs:
    trajs = getdata(label=mol,batch=2000)
    dataset.update(trajs)


reax_nn = ReaxFF_nn(libfile='ffield.json',
                    dataset=dataset,
                    optword='nocoul',mpopt=[1,1,1,1],
                    regularize_mf=1,regularize_be=1,regularize_bias=1,
                    lambda_reg=0.01,lambda_bd=1000.0,lambda_me=0.00001,
                    # mf_universal_nn=['Al','O'],
                    # be_universal_nn=['Al-Al','O-O'],
                    mf_layer=[9,1],be_layer=[9,1],
                    # bdopt= ['O-Al','O-O'],
                    # mfopt=['O'],
                    batch=2000,
                    fixrcbo=False,
                    losFunc='mse',  # n2, mse, huber,abs
                    convergence=0.999) 

reax_nn.initialize()
# GradientDescentOptimizer AdamOptimizer
reax_nn.session(learning_rate=0.0001, method='AdamOptimizer') 

reax_nn.run(learning_rate=1.0e-6,step=0,print_step=1,
              writelib=10,close_session=False)
print(reax_nn.loss_)


bo0 = reax_nn.sess.run(reax_nn.bo0,feed_dict=reax_nn.feed_dict)
#for mol in bo0:
mol = 'AlO-0'
print('\n-  {:s}  -\n'.format(mol))
for b in bo0[mol]:
    # print(b)
    for b_ in b:
        print(b_)

# eb = reax_nn.sess.run(reax_nn.ebond,feed_dict=reax_nn.feed_dict)
# for mol in eb:
#     print('\n-  {:s}  -\n'.format(mol))
#     for e in eb[mol]:
#         print(e)


# print(reax_nn.eover)
# el,dl,nl,d = reax_nn.sess.run([reax_nn.elone,reax_nn.Delta_lp,reax_nn.nlp,
#                               reax_nn.Delta],
#                             feed_dict=reax_nn.feed_dict)
# for mol in el:
#     for i,e in enumerate(el[mol]):
#         #if np.isnan(e):
#         print(e)
#         print(dl[mol][:,i])
#         print(nl[mol][:,i])
#         print(d[mol][:,i])


# (loss,eu,ev,eb,ea,el,ep,ew,tc,
#  e) = reax_nn.sess.run([reax_nn.loss,reax_nn.eunder,reax_nn.eover,
#                                           reax_nn.ebond,reax_nn.eang,
#                                           reax_nn.elone,reax_nn.epen,
#                                           reax_nn.evdw,reax_nn.tconj,
#                                           reax_nn.E],
#                                  feed_dict=reax_nn.feed_dict)
# # print(eo)
# # for mol in e:
# mol = 'AlO-0'
# for i,e_ in enumerate(eu[mol]):
#     print('\n-   {:d}  -\n'.format(i))
#     print('{:s}: '.format(mol),'eunder',e_)
#     print('{:s}: '.format(mol),'eover',ev[mol][i])
#     print('{:s}: '.format(mol),'ebond',eb[mol][i])
#     print('{:s}: '.format(mol),'elone',el[mol][i])
#     print('{:s}: '.format(mol),'eang',ea[mol][i])
#     print('{:s}: '.format(mol),'epen',ep[mol][i])
#     print('{:s}: '.format(mol),'evdw',ew[mol][i])
#     print('{:s}: '.format(mol),'tconj',tc[mol][i])
#     print('{:s}: '.format(mol),'E',e[mol][i])
#     #print('{:s}: '.format(mol),i,loss[mol])

