from os.path import exists,isfile
from os import system,getcwd,chdir
from .prep_data import prep_data


class ColData(object):
  def __init__(self,max_batch=300):
      ''' max_batch: max number of batch 
      '''
      self.max_batch    = max_batch   # max number in direcs to train


  def __call__(self,label=None,dft='ase',batch=50,endstep=None,startstep=0,increase=1):
      self.label  = label
      cwd         = getcwd()
      gen         = self.label+'.gen'
      self.direcs = {}
      i           = startstep
      data_dir    = {}
      running     = True

      traj        = self.label+'.traj'
      if isfile(traj):
         data_dir[self.label] = traj
      else:
         while running:
            run_dir = 'aimd_'+self.label+'/'+self.label+'-'+str(i)
            if exists(run_dir):
                i += increase
                data_dir[self.label+'-'+str(i)] = cwd+'/'+run_dir+'/'+self.label+'.traj'
            else:
                running = False
            if not endstep is None:
                if increase>0:
                    if i>endstep: running = False
                else:
                    if i<endstep: running = False
                    
      trajs_ = prep_data(label=self.label,direcs=data_dir,
                         split_batch=batch,max_batch=self.max_batch,
                         frame=100000,dft=dft)              # get trajs for training
      return trajs_

class ColRawData(object):
  def __init__(self,max_batch=1000):
      ''' max_batch: max number of batch
      '''
      self.max_batch    = max_batch   # max number in direcs to train

  def __call__(self,dft='ase',batch=50,endstep=None,rawdata={}):
      cwd           = getcwd()
      # gen         = self.label+'.gen'
      #self.direcs  = {}
      # i           = startstep
      data      = {}
      # running     = True

      for key in rawdata:
          trajs_ = prep_data(label=key,direcs={key:rawdata[key]},
                             split_batch=batch,max_batch=self.max_batch,
                             frame=100000,dft=dft)              # get trajs for training
          # print(trajs_)
          data.update(trajs_)
      return data

if __name__ == '__main__':
   getdata = ColData(batch=100)
   traj    = getdata(label='nm2')


