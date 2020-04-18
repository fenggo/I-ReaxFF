#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
import paramiko
from os import environ,system,getcwd
from os.path import exists
from .dft.siesta import siesta_md,siesta_opt
from .molecule import compress,press_mol
# from .mdtodata import MDtoData
from ase.io import read,write
from ase import Atoms


class server(object):
   ''' a host for compute '''
   def __init__(self,ip=None,
                username=None,password=None,timeout=20,port=22,
                cmd=None,
                direc=None):
       self.direc = direc
       self.ssh = paramiko.SSHClient()
       self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
       self.ssh.connect(ip,port, username=username, password=password, timeout=timeout)
      
       self.sftp_transport = paramiko.Transport((ip,port))
       self.sftp_transport.connect( username=username,password=password)
       self.sftp = paramiko.SFTPClient.from_transport(self.sftp_transport)
       try:
          self.ssh.chdir(remotedirectory) # sub-directory exists
       except:
          print('-  directory %s already exists.\n' %self.direc)
       try:
          self.ssh.exec_command('rm *')
       except:
          print('-  directory %s is ready.\n' %self.direc)
       self.ssh.chdir(direc)
       self.sftp.chdir(direc)

   def exe_cmd(cmd=None):
       stdin, stdout, stderr = self.ssh.exec_command(cmd)
       return stdout.readlines()

   def get(rf,lf):
       self.sftp.get(remotepath=rf,localpath=lf)

   def put(rf,lp):
       self.sftp.put(remotepath=rf,localpath=lf)

   def close(self):
       self.ssh.close()
       self.sftp_transport.close()


def cmd(ncpu=4,T=350,comp=[0.99,1.0,0.999],us='F'):
    system('rm siesta.MDE siesta.MD_CAR')
    A = read('packed.gen')
    # A = press_mol(A)
    if not comp is None:
       if us=='T':
          fx = open('siesta.XV','r')
          lines = fx.readlines()
          fx.close()
          fx = open('siesta.XV','w')
          for i,line in enumerate(lines):
              if i<3:
                 l = line.split()
                 print(float(l[0])*comp[i],float(l[1])*comp[i],float(l[2])*comp[i],
                       l[3],l[4],l[5],file=fx)
              else:
                 print(line[:-1],file=fx)
          fx.close()
       else:
         A = compress(A,comp=comp)
    siesta_md(A,ncpu=ncpu,T=T,dt=0.1,tstep=2000,us=us)


def md(ncpu=4,T=350,us='F',tstep=51,dt=1.0,gen='packed.gen',server=None):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=-1)
    A = press_mol(A)
    print('\n-  running siesta md ...')
    if server is None:
       siesta_md(A,ncpu=ncpu,T=T,dt=dt,tstep=tstep,us=us)
    else:
       server.exe_cmd('rm siesta.*')
       server.put('press_mol.gen','packed.gen')
       cmd  =  'python -c'
       cmd += '"from irff.siesta import siesta_md;'
       cmd +=  'from ase.io import read;'
       cmd +=  'A=read("packed.gen");'
       cmd +=  'siesta_md(A,ncpu=ncpu,T=T,dt=dt,tstep=tstep,us=us)"'
       server.exe_cmd(cmd)
       server.get('siesta.MDE','siesta.MDE')
       server.get('siesta.MD_CAR','siesta.MD_CAR')


def opt(ncpu=4,T=350,us='F',tstep=501,dt=1.0,gen='packed.gen',server=None):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=-1)
    A = press_mol(A)
    print('\n-  running siesta opt ...')
    if server is None:
       siesta_opt(A,ncpu=ncpu,us=us)
    else:
       server.exe_cmd('rm siesta.*')
       server.put('press_mol.gen','packed.gen')
       cmd  =  'python -c'
       cmd += '"from irff.siesta import siesta_opt;'
       cmd +=  'from ase.io import read;'
       cmd +=  'A=read("packed.gen");'
       cmd +=  'siesta_md(A,ncpu=ncpu,us=us)"'
       server.exe_cmd(cmd)
       server.get('siesta.MDE','siesta.MDE')
       server.get('siesta.MD_CAR','siesta.MD_CAR')
       

if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [md,cmd,opt])
   argh.dispatch(parser)



