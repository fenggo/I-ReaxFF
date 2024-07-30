#!/usr/bin/env python
from ase import Atoms
from irff.reax_nn import ReaxFF_nn
from irff.data.ColData import ColData

getdata = ColData()

dataset = {}
strucs = [#'fox',
          'fox7']

weight  = {'others':2.0}
batchs  = {'others':800}
batch   = 800

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)


clip = {'Desi':(100.0,725.0),
        'bo1':(-0.3,-0.002),'bo2':(4.0,9.9),'bo3':(-0.3,-0.003),'bo4':(4.0,9.9),
        'bo5':(-0.3,-0.003),'bo6':(4.0,9.9),
        'rosi':(0.5,1.5),'ropi':(0.5,1.46),'ropp':(0.5,1.46),
        'ovun1':(0.0,1.0),'ovun2':(-1.20,0.0),'ovun3':(0.0066,6.0),
        'rvdw_C':(1.84,2.4),
        'Devdw':(0.01,0.8),'alfa':(6.0,17.0),
        'vdw1':(0.50,8.0),
        'gammaw':(1.7,14.0),
        'val1':(10,60),'val2':(0.05,1.98),'val3':(0.01,7.6),
        'val4':(0.01,0.698), # cause NaN !!
        'tor1':(-5.0,-0.049),'tor2':(0.41,5.0),'tor3':(0.041,5.0),'tor4':(0.05,1.0),
        'V1':(0.0,20),'V2':(0,48),'V3':(0.0,10),
        'cutoff':(0.0001,0.01),'acut':(0.0010,0.010)}
 
cons = ['lp2',#'lp1',
        'theta0','val8',
        'valang','val9',
        'valboc','cot1','cot2','coa1','coa2','coa3','coa4',
        # 'pen1','pen2','pen3','pen4',
        #'vdw1','gammaw','rvdw','alfa','Devdw',
        'rohb','Dehb','hb1','hb2',
        # 'ovun6','ovun7','ovun8','ovun5',
        'val','lp3',#'cutoff',
        # 'acut',
        ]    # 不进行局域优化
# cons.extend(['Depi','Depp','Desi',
#                 'rosi','ropi','ropp',
#                 'bo1','bo2','bo3','bo4','bo5','bo6']) ### 
# cons.extend(['tor1','tor2','tor3','tor4','V1','V2','V3'])
# cons.extend(['val1','val2','val3','val6','val7','val4','val5']) 
# cons.extend(['theta0','vale','val9','val10','val8','valang']) # 
# cons.extend(['ovun2','ovun3','ovun4','ovun1','ovun6','ovun7','ovun8','ovun5'])
# cons.extend(['vdw1','gammaw','rvdw','alfa','Devdw'])
# cons.extend(['rohb','Dehb','hb1','hb2'])
mf_layer = [9,2]

if __name__ == '__main__':
   ''' run train '''
   # while True:
   rn = ReaxFF_nn(dataset= dataset,# {'gp8':'data/gp8-0.traj'},
                    libfile='ffield.json',
                    MessageFunction=3,
                    mf_layer=mf_layer,
                    be_layer=[9,1],
                    cons=cons,clip=clip,
                    screen=True,
                    fixrcbo=False,lambda_bd=3000.0,
                    lambda_reg=0.003,
                    convergence=0.9999)

   rn.run(learning_rate=0.0001,step=5000,writelib=1000)
    

