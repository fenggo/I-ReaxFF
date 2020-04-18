from os.path import exists
from os import system,getcwd,chdir
from ase.io import read,write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from irff.molecule import packmol
from irff.gmd import nvt_wt as gulp_nvt
from irff.smd import md as siesta_md
from irff.mdtodata import MDtoData



def plot_energies(it,edft,eamp):
    plt.figure()             # test
    plt.ylabel('Energies (eV)')
    plt.xlabel('Iterations')

    plt.plot(it,edft,linestyle='-',marker='o',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='k',
             ms=4,c='k',alpha=0.01,label='SIESTA')

    plt.plot(it,eamp,linestyle='-',marker='^',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='k',
             ms=4,c='r',alpha=0.01,label='ReaxFF')

    plt.legend(loc='best')
    plt.savefig('energies.eps') 
    plt.close() 


def get_zpe(i,p,zpe,atom_name,label):
    zpe_ = 0.0
    if i==0:
       for atom in atom_name:
           zpe_ += p['atomic_'+atom]
    else:
       zpe_ = zpe[label]
    return zpe_
    

def rnn(dirs=None,nbatchs=None,ncpu=12,
        total_step=2,train_step=500,dt=1.0,
        batch=100,step=10,
        label='case2',
        train_direct=True,
        covergence=0.1):
    if train_direct:
       from .train import r as training
    else:
       from .train import t as training

    system('./r2l<ffield>reax.lib')
    direcs = {}
    for mol in dirs:
        nb = nbatchs[mol] if mol in nbatchs else 1
        for i in range(nb):
            direcs[mol+'-'+str(i)] = dirs[mol]

    # training loop
    it      = []
    e_gulp  = []
    e_siesta= []

    do_ = True
    while do_:
        p,zpe = training(direcs=direcs,step=train_step,
        	             batch=batch,total_step=total_step)
        cwd = getcwd()
        system('./r2l<ffield>reax.lib')

        run_dir = 'run_'+str(i)
        if i==0:
           direcs[label] = cwd + '/' + label+'.traj'
        elif i*step%batch==0:
           direcs[label] = cwd + '/' + label+'-'+str(i*step/batch)+'.traj'
        else:
           system('rm '+label+'.pkl')
           if exists(label+'-*.pkl'):
              system('rm '+label+'-*.pkl')

        if exists(label+'.traj'):
           gen = label+'.traj'
        else:
           gen = 'packed.gen'

        gulp_nvt(T=350,time_step=dt,tot_step=step,gen=gen,mode='a')
        atoms_gmd = read('his.traj',index=-1)
        atom_name = atoms_gmd.get_chemical_symbols()
        zpe_ = get_zpe(i,p,zpe,atom_name,label)
        e_gulp.append(atoms_gmd.get_potential_energy())

        if not exists(run_dir):
           system('mkdir '+run_dir)
           chdir(cwd+'/'+run_dir)
           system('cp ../*.psf ./')
           siesta_md(ncpu=ncpu,T=300,dt=dt,us='F',tstep=step,gen= cwd + '/' +'his.traj')
           chdir(cwd)

        d = MDtoData(structure=label,dft='siesta',dic=cwd+'/'+run_dir,batch=step)
        d.get_traj()
        d.close()

        atoms_smd = read('%s.traj' %label,index=-step)
        e_siesta.append(atoms_smd.get_potential_energy()-zpe_)

        if abs(e_siesta[-1]-e_gulp[-1])<covergence:
           do_ = False

        del atoms_smd
        del atoms_gmd
        it.append(i)
        plot_energies(it,e_siesta,e_gulp)


     

