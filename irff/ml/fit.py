#!/usr/bin/env python
from distutils.command.build import build
from os.path import isfile
from os import system
import argh
import argparse
import json as js
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from irff.ml.data import get_data
from irff.data.ColData import ColData
from irff.intCheck import init_bonds
tf.compat.v1.disable_eager_execution()


def resolve():
    res = minimize_scalar(func,method='brent')
    print(res.x)
    t = sigmoid(0.8813735870089523)
    print(t*t)


class Linear_be(object):
    def __init__(self,Bp,D,B,E,be_layer=None,bonds=None,random_init=0):
        self.unit = 4.3364432032e-2
        with open('ffield.json','r') as lf:
            self.j = js.load(lf)
        self.spec,bonds_,offd,angs,torp,hbs = init_bonds(self.j['p'])
        self.bonds = bonds_ if bonds is None else bonds 
        self.m  = {}
        self.De = {}
        self.De_= {}
        hidelayer  = self.j['be_layer'][1] if be_layer is None else be_layer[1]
        self.be_layer = self.j['be_layer'] if be_layer is None else be_layer
        l = self.be_layer

        for bd in E:
            self.De[bd]        = tf.Variable(self.j['p']['Desi_'+bd]*self.unit,name='Desi_'+bd)

        self.E,self.B = {},{}
        for bd in self.bonds:
            # self.De_[bd]     = tf.clip_by_value(self.De[bd],0.0,999*self.unit)
            # self.De[bd]      = tf.constant(self.j['p']['Desi_'+bd]*self.unit,name='Desi_'+bd)
            if random_init:
               self.m['fewi_'+bd] = tf.Variable(tf.random.normal([3,l[0]],stddev=0.1),name='fewi_'+bd)
               self.m['febi_'+bd] = tf.Variable(tf.random.normal([l[0]],stddev=0.1),name='febi_'+bd)
               self.m['fewo_'+bd] = tf.Variable(tf.random.normal([l[0],1],stddev=0.1),name='fewo_'+bd)
               self.m['febo_'+bd] = tf.Variable(tf.random.normal([1],stddev=0.1),name='febo_'+bd)
            else:
               self.m['fewi_'+bd] = tf.Variable(self.j['m']['fewi_'+bd],name='fewi_'+bd)
               self.m['febi_'+bd] = tf.Variable(self.j['m']['febi_'+bd],name='febi_'+bd)
               self.m['fewo_'+bd] = tf.Variable(self.j['m']['fewo_'+bd],name='fewo_'+bd)
               self.m['febo_'+bd] = tf.Variable(self.j['m']['febo_'+bd],name='febo_'+bd)
            self.m['few_'+bd]  = []
            self.m['feb_'+bd]  = []
            for i in range(hidelayer):
                # w = np.array(self.j['m']['few_'+bd][i])
                # b = np.array(self.j['m']['few_'+bd][i])
                # m,n = w.shape
                # print('hidden layer:',i,m,n)
                if i+1 > self.j['be_layer'][1]:
                   self.m['few_'+bd].append(tf.Variable(self.j['m']['few_'+bd][-1],name='fewh_'+bd))
                   self.m['feb_'+bd].append(tf.Variable(self.j['m']['feb_'+bd][-1],name='febh_'+bd))
                   self.j['m']['few_'+bd].append(self.j['m']['few_'+bd][-1])
                   self.j['m']['feb_'+bd].append(self.j['m']['feb_'+bd][-1])
                else:
                   if random_init:
                      self.m['few_'+bd].append(tf.Variable(tf.random.normal([l[0],l[0]],stddev=0.1),
                                               name='fewh_'+bd))
                      self.m['feb_'+bd].append(tf.Variable(tf.random.normal([l[0]],stddev=0.1),
                                               name='febh_'+bd))
                   else:
                      self.m['few_'+bd].append(tf.Variable(self.j['m']['few_'+bd][i],name='fewh_'+bd))
                      self.m['feb_'+bd].append(tf.Variable(self.j['m']['feb_'+bd][i],name='febh_'+bd))


        if hidelayer > self.j['be_layer'][1]:
           self.j['be_layer'][1] = hidelayer
        for bd in self.De:
            self.E[bd] = tf.compat.v1.placeholder(tf.float32,shape=[None,1],name='E_%s' %bd)
            self.B[bd] = tf.compat.v1.placeholder(tf.float32,shape=[None,3],name='B_%s' %bd)
            #print('define the placeholder for the model ...')
        self.loss = self.forward()     
        self.feed_dict = self.feed_data(Bp,D,B,E)

    def forward(self):
        #print('build graph ...')
        loss = 0.0
        self.E_pred = {}
        for bd in self.De:
            ai   = tf.sigmoid(tf.matmul(self.B[bd],self.m['fewi_'+bd])  + self.m['febi_'+bd])
            if self.be_layer[1]>0:
               for i in range(self.be_layer[1]):
                   if i==0:
                      a_ = ai
                   else:
                      a_ = ah
                   ah = tf.sigmoid(tf.matmul(a_,self.m['few_'+bd][i]) + self.m['feb_'+bd][i])
               ao = tf.sigmoid(tf.matmul(ah,self.m['fewo_'+bd]) + self.m['febo_'+bd])
            else:
               ao = tf.sigmoid(tf.matmul(ai,self.m['fewo_'+bd]) + self.m['febo_'+bd])

            self.E_pred[bd] = ao*self.De[bd]
            # loss+= tf.reduce_sum(tf.abs(self.E[bd]-self.E_pred[bd]))
            # loss  += tf.nn.l2_loss(self.E[bd]-self.E_pred[bd])
            loss+= tf.reduce_sum(tf.square(self.E[bd]-self.E_pred[bd]))
            # loss+= tf.compat.v1.losses.absolute_difference(self.E[bd],self.E_pred[bd])
            # loss+= tf.compat.v1.losses.mean_squared_error(self.E[bd],self.E_pred[bd])
        return loss

    def session(self,learning_rate=3.0e-4,method='AdamOptimizer'):
        self.config = tf.compat.v1.ConfigProto()
        self.sess   = tf.compat.v1.Session(config=self.config)  
        optimizer   = tf.compat.v1.train.AdamOptimizer(learning_rate) 
        self.train_step = optimizer.minimize(self.loss)
        self.sess.run(tf.compat.v1.global_variables_initializer())  

    def run(self,learning_rate=1.0e-4,method='AdamOptimizer',step=2000,convergence=0.0001):
        self.session(learning_rate=learning_rate,method=method)  

        for i in range(step+1):
            loss,_ = self.sess.run([self.loss,self.train_step],feed_dict=self.feed_dict)
            
            if i % 10 == 0:
               print('Step: %d Loss=%.8f' %(i,loss))
            if i % 1000 == 0:
               self.save()
            if loss<convergence:
               self.save()
               break

        # B,E,E_pred = self.sess.run([self.B,self.E,self.E_pred],feed_dict=self.feed_dict)
        # for bd in ['Si-Si']:
        #     print('\n bond: {:s} \n'.format(bd))
        #     for b,e,e_pred in zip(B[bd],E[bd],E_pred[bd]):
        #         print('b:',b,'e: ',e,'e_pred: ',e_pred)

    def feed_data(self,Bp,D,B,E):
        feed_dict = {}
        for bd in self.De:
            feed_dict[self.B[bd]] = np.array(B[bd]).astype(np.float32)
            feed_dict[self.E[bd]] = np.expand_dims(E[bd],axis=1).astype(np.float32)
        return feed_dict

    def save(self):
        self.j['EnergyFunction'] = 1
        for bd in self.De:
            self.j['m']['fewi_'+bd] = self.sess.run(self.m['fewi_'+bd]).tolist()
            self.j['m']['febi_'+bd] = self.sess.run(self.m['febi_'+bd]).tolist()
            self.j['m']['fewo_'+bd] = self.sess.run(self.m['fewo_'+bd]).tolist()
            self.j['m']['febo_'+bd] = self.sess.run(self.m['febo_'+bd]).tolist()
            self.j['p']['Desi_'+bd] = self.sess.run(self.De[bd])/self.unit

            for i in range(self.be_layer[1]):
                self.j['m']['few_'+bd][i] = self.sess.run(self.m['few_'+bd][i]).tolist()
                self.j['m']['feb_'+bd][i] = self.sess.run(self.m['feb_'+bd][i]).tolist()
        
        with open('ffield.json','w') as fj:
             js.dump(self.j,fj,sort_keys=True,indent=2)

class Linear_bo(object):
    def __init__(self,Bp,D,B,E,bonds=None,mf_layer=None):
        with open('ffield.json','r') as lf:
            self.j = js.load(lf)
        self.spec,bonds_,offd,angs,torp,hbs = init_bonds(self.j['p'])
        self.bonds = bonds_ if bonds is None else bonds 
        self.D,self.D_t,self.B,self.Bp = {},{},{},{}
        self.m = {}
        self.message_function = self.j['MessageFunction']
        self.mf_layer = self.j['mf_layer'] if mf_layer is None else mf_layer

        for sp in self.spec:
            self.m['fmwi_'+sp] = tf.Variable(self.j['m']['fmwi_'+sp],name='fmwi_'+sp)
            self.m['fmbi_'+sp] = tf.Variable(self.j['m']['fmbi_'+sp],name='fmbi_'+sp)
            self.m['fmwo_'+sp] = tf.Variable(self.j['m']['fmwo_'+sp],name='fmwo_'+sp)
            self.m['fmbo_'+sp] = tf.Variable(self.j['m']['fmbo_'+sp],name='fmbo_'+sp)
            self.m['fmw_'+sp]  = []
            self.m['fmb_'+sp]  = []
            for i in range(self.mf_layer[1]):
                if i+1 > self.j['mf_layer'][1]:
                   self.m['fmw_'+sp].append(tf.Variable(self.j['m']['fmw_'+sp][-1],name='fmwh_'+sp))
                   self.m['fmb_'+sp].append(tf.Variable(self.j['m']['fmb_'+sp][-1],name='fmbh_'+sp))
                   self.j['m']['fmw_'+sp].append(self.j['m']['fmw_'+sp][-1])
                   self.j['m']['fmb_'+sp].append(self.j['m']['fmb_'+sp][-1])
                else:
                   self.m['fmw_'+sp].append(tf.Variable(self.j['m']['fmw_'+sp][i],name='fmwh_'+sp))
                   self.m['fmb_'+sp].append(tf.Variable(self.j['m']['fmb_'+sp][i],name='fmbh_'+sp))

        if self.mf_layer[1] > self.j['mf_layer'][1]:
           self.j['mf_layer'][1] = self.mf_layer[1]

        for bd in self.bonds:
            if self.message_function==1:
               self.D[bd]   = tf.compat.v1.placeholder(tf.float32,shape=[None,7],name='D_%s' %bd)
               self.D_t[bd] = tf.compat.v1.placeholder(tf.float32,shape=[None,7],name='Dt_%s' %bd)
            else:
               self.D[bd]   = tf.compat.v1.placeholder(tf.float32,shape=[None,3],name='D_%s' %bd)
               self.D_t[bd] = tf.compat.v1.placeholder(tf.float32,shape=[None,3],name='Dt_%s' %bd)
            self.B[bd]   = tf.compat.v1.placeholder(tf.float32,shape=[None,3],name='B_%s' %bd)
            self.Bp[bd]  = tf.compat.v1.placeholder(tf.float32,shape=[None,3],name='Bp_%s' %bd)
            #print('define the placeholder for the model ...')
        self.loss = self.forward()     
        self.feed_dict = self.feed_data(Bp,D,B,E)

    def forward(self):
        #print('build graph ...')
        loss = 0.0
        for bd in self.bonds:
            atomi,atomj = bd.split('-')
            ai   = tf.sigmoid(tf.matmul(self.D[bd],self.m['fmwi_'+atomi])  + self.m['fmbi_'+atomi])
            for i in range(self.j['mf_layer'][1]):
                if i==0:
                   a_ = ai
                else:
                   a_ = ah
                ah   = tf.sigmoid(tf.matmul(a_,self.m['fmw_'+atomi][i]) + self.m['fmb_'+atomi][i])
                
            ao   = tf.sigmoid(tf.matmul(ah,self.m['fmwo_'+atomi]) + self.m['fmbo_'+atomi])

            ai_t = tf.sigmoid(tf.matmul(self.D_t[bd],self.m['fmwi_'+atomj]) + self.m['fmbi_'+atomj])
            for i in range(self.j['mf_layer'][1]):
                if i==0:
                   a_ = ai_t
                else:
                   a_ = ah_t
                ah_t  = tf.sigmoid(tf.matmul(a_,self.m['fmw_'+atomj][i]) + self.m['fmb_'+atomj][i])
            #ah_t = tf.sigmoid(tf.matmul(ai_t,self.m['fmw_'+atomj][0]) + self.m['fmb_'+atomj][0])
            ao_t = tf.sigmoid(tf.matmul(ah_t,self.m['fmwo_'+atomj]) + self.m['fmbo_'+atomj])
            
            if self.message_function==2:
               b_pred = ao*ao_t
            else:
               b_pred = self.Bp[bd]*ao*ao_t
            loss+= tf.sqrt(tf.reduce_sum(tf.square(self.B[bd]-b_pred)))
            # loss+= tf.nn.l2_loss(self.B[bd]-b_pred)
            # loss+= tf.compat.v1.losses.absolute_difference(self.B[bd]-b_pred)
        return loss

    def session(self,learning_rate=3.0-4,method='AdamOptimizer'):
        self.config = tf.compat.v1.ConfigProto()
        self.sess   = tf.compat.v1.Session(config=self.config)  
        optimizer   = tf.compat.v1.train.AdamOptimizer(learning_rate) 
        self.train_step = optimizer.minimize(self.loss)
        self.sess.run(tf.compat.v1.global_variables_initializer())  

    def run(self,learning_rate=1.0e-4,method='AdamOptimizer',step=2000,convergence=0.0001):
        self.session(learning_rate=learning_rate,method=method)  

        for i in range(step+1):
            loss,_ = self.sess.run([self.loss,self.train_step],feed_dict=self.feed_dict)
            
            if i % 10 == 0:
               print('Step: %d Loss=%.8f' %(i,loss))
            if i % 1000 == 0:
               self.save()
            if loss<convergence:
               self.save()
               break

    def feed_data(self,Bp,D,B,E):
        feed_dict = {}
        for bd in Bp:
            d = np.array(D[bd]).astype(np.float32)
            feed_dict[self.D[bd]]   = d
            if self.message_function==1:
               feed_dict[self.D_t[bd]] = d[:,[6,5,4,3,2,1,0]]
            else:
               feed_dict[self.D_t[bd]] = d[:,[2,1,0]]
            feed_dict[self.B[bd]]   = np.array(B[bd]).astype(np.float32)
            feed_dict[self.Bp[bd]]  = np.array(Bp[bd]).astype(np.float32)
        return feed_dict

    def save(self):
        for sp in self.spec:
            self.j['m']['fmwi_'+sp] = self.sess.run(self.m['fmwi_'+sp]).tolist()
            self.j['m']['fmbi_'+sp] = self.sess.run(self.m['fmbi_'+sp]).tolist()
            self.j['m']['fmwo_'+sp] = self.sess.run(self.m['fmwo_'+sp]).tolist()
            self.j['m']['fmbo_'+sp] = self.sess.run(self.m['fmbo_'+sp]).tolist()

            for i in range(self.j['mf_layer'][1]):
                self.j['m']['fmw_'+sp][i] = self.sess.run(self.m['fmw_'+sp][i]).tolist()
                self.j['m']['fmb_'+sp][i] = self.sess.run(self.m['fmb_'+sp][i]).tolist()

        with open('ffield.json','w') as fj:
             js.dump(self.j,fj,sort_keys=True,indent=2)


def train(Bp,D,B,E,convergence=0.000001,
          learning_rate=0.01,layer=None,random_init=0,
          step=5000,fitobj='BO',bonds=None):
    # neural network layers
    if fitobj == 'BO':
       model = Linear_bo(Bp,D,B,E,bonds=bonds,mf_layer=layer)
    elif fitobj == 'BE':
       model = Linear_be(Bp,D,B,E,bonds=bonds,be_layer=layer,random_init=random_init)

    model.run(step=step,convergence=convergence,learning_rate=learning_rate)


def fit(step=1000,obj='BO'):
    dataset = {'dia-0':'data/dia-0.traj',
            'dia-1':'data/dia-1.traj',
            'dia-2':'data/dia-2.traj',
            # 'dia-3':'data/dia-3.traj',
            'gp2-0':'data/gp2-0.traj',
            'gp2-1':'data/gp2-1.traj',
            #'gpd-0':'data/gpd-0.traj',
            #'gpd-1':'data/gpd-1.traj',
            #'gpd-2':'data/gpd-2.traj',
            #'gpd-3':'data/gpd-3.traj',
            #'gpd-4':'data/gpd-4.traj',
            #'gpd-5':'data/gpd-5.traj',
            #'gpd-6':'data/gpd-6.traj',
            #'gpd-7':'data/gpd-7.traj',
            #'gpd-8':'data/gpd-8.traj',
            #'gpd-9':'data/gpd-9.traj',
            }

    trajdata   = ColData()
    strucs = [#'c32',
             #'c6',
             #'c10',
             #'ch4',
             ]
    batchs = {'others':50}

    for mol in strucs:
        b = batchs[mol] if mol in batchs else batchs['others']
        trajs = trajdata(label=mol,batch=b)
        dataset.update(trajs)

    bonds = ['C-C']
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    for bd in B:
        # print(len(B[bd]))
        B[bd] =np.array(B[bd])/1.2

    train(Bp,D,B,E,step=step,fitobj=obj)


if __name__ == '__main__':
   ''' Run with commond: ./gpfit.py fit --o=BE --s=3000 '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [fit])
   argh.dispatch(parser)  



