import json as js
import numpy as np
from irff.intCheck import init_bonds

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

class fnn(object):
    def __init__(self,ffield='ffieldData.json'):
        with open(ffield,'r') as lf:
            self.j = js.load(lf)
        self.spec,self.bonds,offd,angs,torp,hbs = init_bonds(self.j['p'])
        self.m = {}
        hidelayer  = self.j['be_layer'][1] 
        self.be_layer = self.j['be_layer'] 

        self.E_pred = {}
        self.B_pred = {}
        
        for bd in self.bonds:
            self.m['fewi_'+bd] = self.j['m']['fewi_'+bd]
            self.m['febi_'+bd] = self.j['m']['febi_'+bd]
            self.m['fewo_'+bd] = self.j['m']['fewo_'+bd]
            self.m['febo_'+bd] = self.j['m']['febo_'+bd]
            self.m['few_'+bd]  = []
            self.m['feb_'+bd]  = []
            for i in range(hidelayer):
                self.m['few_'+bd].append(self.j['m']['few_'+bd][i])
                self.m['feb_'+bd].append(self.j['m']['feb_'+bd][i])

        for sp in self.spec:
            self.m['fmwi_'+sp] = self.j['m']['fmwi_'+sp]
            self.m['fmbi_'+sp] = self.j['m']['fmbi_'+sp]
            self.m['fmwo_'+sp] = self.j['m']['fmwo_'+sp]
            self.m['fmbo_'+sp] = self.j['m']['fmbo_'+sp]
            self.m['fmw_'+sp]  = []
            self.m['fmb_'+sp]  = []
            for i in range(self.j['mf_layer'][1]):
                self.m['fmw_'+sp].append(self.j['m']['fmw_'+sp][i])
                self.m['fmb_'+sp].append(self.j['m']['fmb_'+sp][i])

    def compute_bond_energy(self,B):
        self.B      = B
        for bd in self.B:
            ai   = sigmoid(np.matmul(self.B[bd],self.m['fewi_'+bd])  + self.m['febi_'+bd])
            if self.be_layer[1]>0:
               for i in range(self.be_layer[1]):
                   if i==0:
                      a_ = ai
                   else:
                      a_ = ah
                   ah = sigmoid(np.matmul(a_,self.m['few_'+bd][i]) + self.m['feb_'+bd][i])
               ao = sigmoid(np.matmul(ah,self.m['fewo_'+bd]) + self.m['febo_'+bd])
            else:
               ao = sigmoid(np.matmul(ai,self.m['fewo_'+bd]) + self.m['febo_'+bd])

            self.E_pred[bd] = ao
            # loss  += tf.nn.l2_loss(self.E[bd]-self.E_pred[bd])
        return self.E_pred
    
    def compute_bond_order(self,Bp,D):
        self.D      = D
        for bd in self.D:
            atomi,atomj = bd.split('-')
            ai   = sigmoid(np.matmul(self.D[bd],self.m['fmwi_'+atomi])  + self.m['fmbi_'+atomi])
            for i in range(self.j['mf_layer'][1]):
                if i==0:
                   a_ = ai
                else:
                   a_ = ah
                ah   = sigmoid(np.matmul(a_,self.m['fmw_'+atomi][i]) + self.m['fmb_'+atomi][i])
                
            ao   = sigmoid(np.matmul(ah,self.m['fmwo_'+atomi]) + self.m['fmbo_'+atomi])

            D_t  = np.array(self.D[bd])[:,[2,1,0]]

            ai_t = sigmoid(np.matmul(D_t,self.m['fmwi_'+atomj]) + self.m['fmbi_'+atomj])
            for i in range(self.j['mf_layer'][1]):
                if i==0:
                   a_ = ai_t
                else:
                   a_ = ah_t
                ah_t  = sigmoid(np.matmul(a_,self.m['fmw_'+atomj][i]) + self.m['fmb_'+atomj][i])
            ao_t = sigmoid(np.matmul(ah_t,self.m['fmwo_'+atomj]) + self.m['fmbo_'+atomj])

            b_pred = Bp[bd]*ao*ao_t
            self.B_pred[bd] = b_pred
        return self.B_pred


# from irff.ml.nn import fnn
# nn = fnn()
# bo = nn.compute_bond_order({bd:[b]},{bd:[d]})

# be = nn.compute_bond_energy(bo)
# be_ = np.squeeze(be[bd])
# if be_>=0.0:
#    print('\nnn input(D)    : \n',d)
#    print(  'nn predict (bo): ',bo[bd][0])
#    print(  'nn predict (E) : ',be_)

