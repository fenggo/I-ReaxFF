'''
IRFF: Intelligent Reactive Force Filed, A ReaxFF developing tool kit.
Authored by FengGo.
email: fengguo@lcu.edu.cn
       gfeng.alan@foxmail.com

This software package is under academic license.
  2022-06-30:
     version 1.4.0 use Di-Bij and Dj-Bij instead of Di and Dj as input vector.
  2022-06-24:
     version 1.3.9 add elementary message function, and implement in the GULP Program.
  2022-06-01:
     version 1.3.8 add f4-->fnn, and implement the f1 correction  
  2022-01-17:
     version 1.3.3  
  2021-05-31:
     version 1.2.4 All bond share the same neural networks weights and biases.
  2021-05-28:
     version 1.2.3 Regularization algriothm is implemented to overcome the overfit problem.
  2021-05-11:
     version 1.2.0 A new restricted molecular dynamic simulation methond has implemented to 
  seach the configration space. 
  2020-02-01:
     version 0.8 Massage passing mode  introduced
  2019-11-01:
     version 0.7 Support TensorFlow-2.0
  2019-09-18:
     version 0.6 supporting for triclinic crystal strucutres
  2019-08-05:
     version 0.5 opitimize ReaxFF-QEq parameters with machine learning
     
  2019-05-06:
     version 0.3 fix bugs of NLP, PBO calculation 
                          Delta_angle index error
  2019-04-16
     version 0.2 weight added
  2019-04-02:
     version 0.1 change log, test of torsion energies,
     found eover and eunder bugs(fixed)
     add penalty for case of bond-order to small

ooooo      ooooo     oooooo     ooo    o       o   oooooo   oooooo
  o        o    o    o         o   o    o     o    o        o
  o        o     o   o        o     o    o   o     o        o
  o   ooo  oooooo    oooooo   ooooooo     ooo      o        o
  o        o   o     o        o     o      o       oooooo   oooooo
  o        o    o    o        o     o     ooo      o        o
  o        o     o   o        o     o    o   o     o        o
ooooo      o      o  oooooo   o     o   o     o    o        o
'''


'''
variables that cause NaN errors:
val2 68 --> 6, 2019,05,29
val7 25 --> 15, 2019,05,29
val6 15 --> 5
val3 24 --> 14
ovun4 7 --> 5
ovun8 9-->3   2.89 --> 1.89
tor3 16 --> 14
tor2 18 --> 2
tor4 20 --> 2
boc1 25 --> 20
lp2  48 --> 8
bo2  48 --> 8
boc3 41 --> 1.0
pen3 28 --> 8   2019,11,22
pen4 13 --> 5
vdw1 29 --> 2
coa2 29 --> 2
'''

# from os import getcwd
# print(getcwd())

