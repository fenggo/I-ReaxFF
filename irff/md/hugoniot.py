from __future__ import print_function



class Hugoniostat(object):
  def __init__(self,pressure=None,dT=5.0,ncpu=1,gen='poscar.gen',
               skf_dir='./',
               maxam = {'C':'p','H':'s','O':'p','N':'p'},
               hubbard={'C':-0.1492,'H': -0.1857,'O':-0.1575,'N': -0.1535}):
      '''
      Hugoniot equation of state.
      pressure = [0.001,2,4,6,8,10,12,14,16,18,20,22]
      '''
      # self.pressure  = np.linspace(0.0001, 20, 20)
      self.pressure = pressure
      self.np       = ncpu
      self.dT       = dT
      self.gen      = gen
      self.skf_dir  = skf_dir
      self.maxam    = maxam
      self.hubbard  = hubbard


  def hugoniot(self,totalstep=20000):
      N,t0,p0,e0,v0 = self.N,self.t0,self.p0,self.e0,self.v0

      print("****************************************************************")
      print("* Starting point:                                              *")
      print("* Pressure: %f, Energy     : %f         *"  %(p0,e0))
      print("* Volume  : %f,  Temperature: %f            *" %(v0,t0))
      print("* Free particals number: %d                                  *" %N)
      print("****************************************************************")
      print('* *')
      print('* *')

      if t0 <= 0:
         T = 0.01
      else:
         T = t0
      system('cp restart.eq restart.x')

      for i in self.pressure:
          P =  i*10000
          dT = 11.0
          while dT>self.dT or dT<-self.dT:
             if T<=0.0:
                T = 1.0
             print('* ')
             print('****************************** target pressure ******************************')
             print('**')
             print('**                   target Pressure now: %12.5f GPa                 **' %(P*1.01325/10000))
             print('**')
             print('****************************** target pressure ******************************')
             print('* ')

             # send_msg('-  target Pressure now: %12.5f GPa' %(P*1.01325/10000))

             self.write_lammps_in(log='lmp.log',total=totalstep, restart = 'restart.x',
                  pair_coeff = pair_coeff,
                  pair_style = pair_style,
                  fix = 'fix myhug all nphug temp %f %f 500.0 iso %f %f 500.0' %(T,T,P,P), #Atmospheric pressure
                  fix_modify = 'fix_modify myhug e0 %f p0 %f v0 %f' %(e0, p0, v0) ,
                  more_commond = more_commond,
                  thermo_style ='thermo_style     custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz v_tau lz f_myhug v_dele v_us v_up',
                  restartfile = 'restart.x')
             system('cp inp-lam hug.inp')
             self.run_lammps(inp_name='hug.inp',label='hug')
             system('cp lammps.trj hug_%d.lammpstrj' %i)
             step,N,P,V,T,E,us,up,a,b,c,alf,bet,gam=self.get_lammps_hug(logname='lmp.log')      ### get hugoniot datas
             system('mv lmp.log hug_%d.log' %i)
             fhug=open('hug.txt','a')
             flg=open('hlog.txt','a')

             dT = 0.5*(P+p0)*(v0-V)*1.01325*1.0e-25+(e0-E)*6.9477*1.0e-21
             dT = dT/((3*N-3)*1.381*1.0e-23) #the devation to Hugoniostat

             print('*************************** pressure averaged from log file ***************************')
             print('* Pressure: %f, Energy: %f Volume: %f, Temperature: %f, Deviation: %f, ShockVelocity: %f, ParticalVelecity: %f'  
                    %(P*0.0001,E,V,T,dT,us,up))
             print('* Pressure: %f, Energy: %f Volume: %f, Temperature: %f, Deviation: %f, ShockVelocity: %f, ParticalVelecity: %f' 
                    %(P*0.0001,E,V,T,dT,us,up), file=flg)
             print('*************************** pressure averaged from log file ***************************')
             print(P*0.0001,V,T,E,dT,us,up,a,b,c,alf,bet,gam, file=fhug)
             fhug.close()
             flg.close()

             self.gplot('upus.eps','hug.txt',axis=[(6,7)],xlab='Partical Velocity',ylab='Shock Velocity',title=['Partical-Shock'])
             self.gplot('dev.eps','hug.txt',axis=[(1,5)],xlab='Pressure',ylab='Deviation',title=['Deviation'])
             self.gplot('pv.eps','hug.txt',axis=[(1,2)],xlab='Pressure (GPa)',ylab='Volume',title=['Volume'])
             self.gplot('upus.png','hug.txt',axis=[(6,7)],xlab='Partical Velocity',ylab='Shock Velocity',title=['Partical-Shock'])
             self.gplot('dev.png','hug.txt',axis=[(1,5)],xlab='Pressure',ylab='Deviation',title=['Deviation'])
             self.gplot('pv.png','hug.txt',axis=[(1,2)],xlab='Pressure (GPa)',ylab='Volume',title=['Volume'])
             T += dT
             
             
