'''
IRFF: Intelligent Reactive Force Filed, A ReaxFF developing tool kit.
Authored by FengGo.
email: fengguo@lcu.edu.cn
       gfeng.alan@foxmail.com

This software package is under academic license.

  2019-09-18:
     version 3.6 supporting for triclinic crystal strucutres
  2019-08-05:
     version 3.5 opitimize ReaxFF-QEq parameters with machine learning
     
  2019-05-06:
     version 3.3 fix bugs of NLP, PBO calculation 
                          Delta_angle index error
  2019-04-16
     version 3.2 weight added
  2019-04-02:
     version 3.1 change log, test of torsion energies,
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
'''

