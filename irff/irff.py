import json as js
import ctypes
import random

import os
import shlex
import shutil
import warnings
from re import IGNORECASE
from re import compile as re_compile
import subprocess
from tempfile import mktemp as uns_mktemp
from threading import Thread
from typing import Any, Dict
import numpy as np

from numpy.linalg import norm

from ase.parallel import paropen
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lammps import (CALCULATION_END_MARK, Prism, convert,
                                    write_lammps_in)
from ase.data import atomic_masses, chemical_symbols, atomic_numbers
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump

# from .qeq import qeq
from .RadiusCutOff import setRcut
from .reaxfflib import read_ffield,write_lib,write_ffield
from .intCheck import init_bonds
from .irff_np import IRFF_NP
# from .md.lammps import writeLammpsIn
# from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb

def is_upper_triangular(arr, atol=1e-8):
    """test for upper triangular matrix based on numpy"""
    # must be (n x n) matrix
    assert len(arr.shape) == 2
    assert arr.shape[0] == arr.shape[1]
    return np.allclose(np.tril(arr, k=-1), 0., atol=atol) and \
        np.all(np.diag(arr) >= 0.0)

def convert_cell(ase_cell):
    """
    Convert a parallelepiped (forming right hand basis)
    to lower triangular matrix LAMMPS can accept. This
    function transposes cell matrix so the bases are column vectors
    """
    cell = ase_cell.T

    if not is_upper_triangular(cell):
        # rotate bases into triangular matrix
        tri_mat = np.zeros((3, 3))
        A = cell[:, 0]
        B = cell[:, 1]
        C = cell[:, 2]
        tri_mat[0, 0] = norm(A)
        Ahat = A / norm(A)
        AxBhat = np.cross(A, B) / norm(np.cross(A, B))
        tri_mat[0, 1] = np.dot(B, Ahat)
        tri_mat[1, 1] = norm(np.cross(Ahat, B))
        tri_mat[0, 2] = np.dot(C, Ahat)
        tri_mat[1, 2] = np.dot(C, np.cross(AxBhat, Ahat))
        tri_mat[2, 2] = norm(np.dot(C, AxBhat))

        # create and save the transformation for coordinates
        volume = np.linalg.det(ase_cell)
        trans = np.array([np.cross(B, C), np.cross(C, A), np.cross(A, B)])
        trans /= volume
        coord_transform = np.dot(tri_mat, trans)

        return tri_mat, coord_transform
    else:
        return cell, None

def rtaper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r<rmin,1.0,0.0) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,0.0)
    r20   = np.where(ok,1.0,0.0)

    rterm = np.divide(1.0,np.power(rmax-rmin,3))
    rm    = rmax*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmin*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3

def taper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r>rmax,1.0,0.0) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,0.0)
    r20   = np.where(ok,1.0,0.0)

    rterm = np.divide(1.0,np.power(rmin-rmax,3)+0.0000001)
    rm    = rmin*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmax*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3

def fvr(x):
    xi  = np.expand_dims(x,axis=0)
    xj  = np.expand_dims(x,axis=1) 
    vr  = xj - xi
    return vr

def fr(vr):
    R   = np.sqrt(np.sum(vr*vr,axis=2))
    return R

def sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s

def relu(x):
    return np.maximum(0, x)


class IRFF(Calculator,IRFF_NP):
    '''Intelligent Machine-Learning ASE calculator based on LAMMPS Python module
       Modified from ASE LAMMPSlib.py
    '''
    name = "IRFF"
    implemented_properties = ["energy", "free_energy", "forces", # "stress",
                              "energies"]

    # parameters to choose options in LAMMPSRUN
    ase_parameters: Dict[str, Any] = dict(
        specorder=None,
        atorder=True,
        always_triclinic=False,
        reduce_cell=False,
        keep_alive=True,
        keep_tmp_files=True,
        no_data_file=False,
        tmp_dir='./',
        files=[],  # usually contains potential parameters
        verbose=False,
        write_velocities=False,
        binary_dump=True,  # bool - use binary dump files (full
                           # precision but long long ids are casted to
                           # double)
        lammps_options="-echo log -screen none -log /dev/stdout",
        trajectory_out=None,  # file object, if is not None the
                              # trajectory will be saved in it
    )

    # parameters forwarded to LAMMPS
    lammps_parameters = dict(
        boundary=None,  # bounadry conditions styles
        units="real",  # str - Which units used; some potentials
                        # require certain units
        atom_style="charge",
        special_bonds=None,
        # potential informations
        pair_style="reaxff control nn yes checkqeq yes",
        pair_coeff=None,
        masses=None,
        pair_modify=None,
        # variables controlling the output
        thermo_args=[
            "step", "temp", "press", "cpu", "pxx", "pyy", "pzz",
            "pxy", "pxz", "pyz", "ke", "pe", "etotal", "vol", "lx",
            "ly", "lz", "atoms", ],
        dump_properties=["id", "type", "x", "y", "z", "vx", "vy",
                         "vz", "fx", "fy", "fz", ],
        dump_period=1,  # period of system snapshot saving (in MD steps)
    )

    default_parameters = dict(ase_parameters, **lammps_parameters)

    def __init__(self,
                 mol=None,atoms=None,
                 libfile='ffield.json',
                 vdwcut=10.0,
                 hbshort=6.75,hblong=7.5,
                 label="IRFF", 
                 active_learning=False,
                 *args,**kwargs):
        Calculator.__init__(self,label=label, **kwargs)
        IRFF_NP.__init__(self,atoms=atoms,nn=True,wf=True,libfile=libfile)

        self.prism  = None
        self.calls  = 0
        self.forces = None
        # thermo_content contains data "written by" thermo_style.
        # It is a list of dictionaries, each dict (one for each line
        # printed by thermo_style) contains a mapping between each
        # custom_thermo_args-argument and the corresponding
        # value as printed by lammps. thermo_content will be
        # re-populated by the read_log method.
        self.thermo_content = []

        symbols = atoms.get_chemical_symbols()
        self.species = sorted(set(symbols))
        sp      = ' '.join(self.species)
        self.parameters['pair_coeff'] = ['* * ffield {:s}'.format(sp)]
        self.masses = {s:atomic_masses[atomic_numbers[s]] for s in self.species } 

        if self.parameters['tmp_dir'] is not None:                   ## del
           # If tmp_dir is pointing somewhere, don't remove stuff!
           self.parameters['keep_tmp_files'] = True
        self._lmp_handle = None  # To handle the lmp process

        self.active_learning = active_learning

    def get_lammps_command(self):
        cmd = self.parameters.get('command')

        if cmd is None:
           cmd = 'lammps -i in.lammps'

        opts = self.parameters.get('lammps_options')

        if opts is not None:
            cmd = f'{cmd} {opts}'
        return cmd

    def clean(self, force=False):
        self._lmp_end()
        if not self.parameters['keep_tmp_files'] or force:
            shutil.rmtree(self.parameters['tmp_dir'])

    def check_state(self, atoms, tol=1.0e-10):
        # Transforming the unit cell to conform to LAMMPS' convention for
        # orientation (c.f. https://lammps.sandia.gov/doc/Howto_triclinic.html)
        # results in some precision loss, so we use bit larger tolerance than
        # machine precision here.  Note that there can also be precision loss
        # related to how many significant digits are specified for things in
        # the LAMMPS input file.
        return Calculator.check_state(self, atoms, tol)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes
        Calculator.calculate(self, atoms, properties, system_changes)
        self.get_bond_energy(atoms=atoms)
        self.run()

    def _lmp_alive(self):
        # Return True if this calculator is currently handling a running
        # lammps process
        return self._lmp_handle and not isinstance(
            self._lmp_handle.poll(), int
        )

    def _lmp_end(self):
        # Close lammps input and wait for lammps to end. Return process
        # return value
        if self._lmp_alive():
            # !TODO: handle lammps error codes
            try:
                self._lmp_handle.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self._lmp_handle.kill()
                self._lmp_handle.communicate()
            err = self._lmp_handle.poll()
            assert err is not None
            return err

    def set_missing_parameters(self):
        """Verify that all necessary variables are set.
        """
        symbols = self.atoms.get_chemical_symbols()
        # If unspecified default to atom types in alphabetic order
        if not self.parameters.get('specorder'):
            self.parameters['specorder'] = sorted(set(symbols))

        # !TODO: handle cases were setting masses actual lead to errors
        if not self.parameters.get('masses'):
            self.parameters['masses'] = []
            for type_id, specie in enumerate(self.parameters['specorder']):
                mass = atomic_masses[chemical_symbols.index(specie)]
                self.parameters['masses'] += [
                    f"{type_id + 1:d} {mass:f}"
                ]

        # set boundary condtions
        if not self.parameters.get('boundary'):
            b_str = " ".join(["fp"[int(x)] for x in self.atoms.pbc])
            self.parameters['boundary'] = b_str

    def run(self, set_atoms=False):
        # !TODO: split this function
        """Method which explicitly runs LAMMPS."""
        pbc = self.atoms.get_pbc()
        if all(pbc):
            cell = self.atoms.get_cell()
        elif not any(pbc):
            # large enough cell for non-periodic calculation -
            # LAMMPS shrink-wraps automatically via input command
            #       "periodic s s s"
            # below
            cell = 2 * np.max(np.abs(self.atoms.get_positions())) * np.eye(3)
        else:
            warnings.warn(
                "semi-periodic ASE cell detected - translation "
                + "to proper LAMMPS input cell might fail"
            )
            cell = self.atoms.get_cell()
        self.prism = Prism(cell)

        self.set_missing_parameters()
        self.calls += 1

        # change into subdirectory for LAMMPS calculations
        tempdir = self.parameters['tmp_dir']

        # setup file names for LAMMPS calculation
        label = f"{self.label}{self.calls:>06}"
        lammps_in = 'in.lammps'
        lammps_log = 'log.lammps'
        lammps_trj = 'lammps.trj'
        if self.parameters['no_data_file']:
            lammps_data = None
        else:
            lammps_data = 'data.lammps'
            write_lammps_data(
                lammps_data,
                self.atoms,
                specorder=self.parameters['specorder'],
                force_skew=self.parameters['always_triclinic'],
                # reduce_cell=self.parameters['reduce_cell'],
                velocities=self.parameters['write_velocities'],
                prismobj=self.prism,
                units=self.parameters['units'],
                atom_style=self.parameters['atom_style'],
            )

            #lammps_data = lammps_data_fd.name
            #lammps_data_fd.flush()

        # see to it that LAMMPS is started
        if not self._lmp_alive():
            command = self.get_lammps_command()
            # Attempt to (re)start lammps
            self._lmp_handle = subprocess.Popen(
                shlex.split(command, posix=(os.name == "posix")),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                encoding='ascii',
            )
        lmp_handle = self._lmp_handle

        # Create thread reading lammps stdout (for reference, if requested,
        # also create lammps_log, although it is never used)

        fd = lmp_handle.stdout
        thr_read_log = Thread(target=self.read_lammps_log, args=(fd,))
        thr_read_log.start()

        # write LAMMPS input (for reference, also create the file lammps_in,
        # although it is never used)
        # print(lmp_handle.stdin)
        fd = 'in.lammps'

        writeLammpsIn(log='/dev/stdout',timestep=0.1,total=0,
              species=self.species,
              masses=self.masses,
              pair_style= self.parameters['pair_style'],  # without lg set lgvdw no
              pair_coeff=self.parameters['pair_coeff'],
              fix = 'fix    fix_nve all nve',
              natoms=len(self.atoms),
              fix_modify = ' ',
              dump_interval=1,more_commond = ' ',
              thermo_style ='thermo_style custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms',
              dump='all custom 1 lammps.trj id type x y z vx vy vz fx fy fz',
              units=self.parameters['units'],
              atom_style=self.parameters['atom_style'],
              data=lammps_data,
              clear='clear')
        # Wait for log output to be read (i.e., for LAMMPS to finish)
        # and close the log file if there is one
        thr_read_log.join()

        if not self.parameters['keep_alive']:
            self._lmp_end()

        exitcode = lmp_handle.poll()
        if exitcode and exitcode != 0:
            raise RuntimeError(
                "LAMMPS exited in {} with exit code: {}."
                "".format(tempdir, exitcode)
            )

        # A few sanity checks
        if len(self.thermo_content) == 0:
            raise RuntimeError("Failed to retrieve any thermo_style-output")
        if int(self.thermo_content[-1]["atoms"]) != len(self.atoms):
            # This obviously shouldn't happen, but if prism.fold_...() fails,
            # it could
            raise RuntimeError("Atoms have gone missing")

        trj_atoms = read_lammps_dump(
            infileobj=lammps_trj,
            order=self.parameters['atorder'],
            index=-1,
            prismobj=self.prism,
            specorder=self.parameters['specorder'],
        )

        if set_atoms:
            self.atoms = trj_atoms.copy()

        self.forces = trj_atoms.get_forces()
        
        # !TODO: trj_atoms is only the last snapshot of the system; Is it
        #        desirable to save also the inbetween steps?
        if self.parameters['trajectory_out'] is not None:
            # !TODO: is it advisable to create here temporary atoms-objects
            self.trajectory_out.write(trj_atoms)

        tc = self.thermo_content[-1]
        self.results["energy"] = convert(
            tc["pe"], "energy", self.parameters["units"], "ASE"
        )
        self.results["free_energy"] = self.results["energy"]
        self.results['forces'] = convert(self.forces.copy(),
                                         'force',
                                         self.parameters['units'],
                                         'ASE')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._lmp_end()

    def read_lammps_log(self, fileobj):
        # !TODO: somehow communicate 'thermo_content' explicitly
        """Method which reads a LAMMPS output log file."""

        # read_log depends on that the first (three) thermo_style custom args
        # can be capitalized and matched against the log output. I.e.
        # don't use e.g. 'ke' or 'cpu' which are labeled KinEng and CPU.
        mark_re = r"^\s*" + r"\s+".join(
            [x.capitalize() for x in self.parameters['thermo_args'][0:3]]
        )
        
        _custom_thermo_mark = re_compile(mark_re)

        # !TODO: regex-magic necessary?
        # Match something which can be converted to a float
        f_re = r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|nan|inf))"
        n_args = len(self.parameters["thermo_args"])
        # Create a re matching exactly N white space separated floatish things
        _custom_thermo_re = re_compile(
            r"^\s*" + r"\s+".join([f_re] * n_args) + r"\s*$", flags=IGNORECASE
        )

        thermo_content = []
        line = fileobj.readline()
        # print('read lammps log ...\n',line)
        while line and line.strip() != CALCULATION_END_MARK:
            # check error
            if 'ERROR:' in line:
                raise RuntimeError(f'LAMMPS exits with error message: {line}')

            # get thermo output
            if _custom_thermo_mark.match(line):
                while True:
                    line = fileobj.readline()
                    if 'WARNING:' in line:
                        continue

                    bool_match = _custom_thermo_re.match(line)
                    if not bool_match:
                        break

                    # create a dictionary between each of the
                    # thermo_style args and it's corresponding value
                    thermo_content.append(
                        dict(
                            zip(
                                self.parameters['thermo_args'],
                                map(float, bool_match.groups()),
                            )
                        )
                    )
            else:
                line = fileobj.readline()

        self.thermo_content = thermo_content
        lines=fileobj.readlines()
        if not thermo_content:
            print('error encoutered when read lammps log: \n')
            for l in lines:
                print(l,end=' ')

    def get_bond_energy(self,atoms=None):
        cell      = atoms.get_cell()                    # cell is object now
        cell      = cell[:].astype(dtype=np.float32)
        rcell     = np.linalg.inv(cell).astype(dtype=np.float32)

        positions = atoms.get_positions()
        xf        = np.dot(positions,rcell)
        xf        = np.mod(xf,1.0)
        positions = np.dot(xf,cell).astype(dtype=np.float32)
        self.get_ebond(cell,rcell,positions)
            

def writeLammpsIn(log='lmp.log',timestep=0.1,total=200, data=None,restart=None,
              species=['C','H','O','N'],
              bond_cutoff={'H-H':1.2,'H-C':1.6,'H-O':1.6,'H-N':1.6,
                           'C-C':2.0,'other':2.0},
              pair_coeff ='* * ffield C H O N',
              pair_style = 'reaxff control nn yes checkqeq yes',  # without lg set lgvdw no
              fix = 'fix   1 all npt temp 800 800 100.0 iso 10000 10000 100',
              fix_modify = ' ',
              more_commond = ' ',
              dump_interval=10,
              freeatoms=None,natoms=None,
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              restartfile=None,
              **kwargs):
    '''
        pair_style     reaxff control.reax checkqeq yes
        pair_coeff     * * ffield.reax.rdx C H O N
        --- control ---
        tabulate_long_range	0 ! denotes the granularity of long range tabulation, 0 means no tabulation
        nbrhood_cutoff		3.5  ! near neighbors cutoff for bond calculations
        hbond_cutoff		7.5  ! cutoff distance for hydrogen bond interactions
        bond_graph_cutoff	0.3  ! bond strength cutoff for bond graphs
        thb_cutoff		    0.001 ! cutoff value for three body interactions
        nnflag              1    ! 0: do not use neural network potential
        mflayer_m           9
        mflayer_n           1
        belayer_m           9
        belayer_n           1
    '''
    random.seed()
    species_name = {'H':'hydrogen','O':'oxygen','N': 'nitrogen','C':'carbon'}
    fin = open('in.lammps','w')
 
    if 'clear' in kwargs:
       print('clear   \n', file=fin)

    if 'units' in kwargs:
       units = kwargs['units']
       print('units     {:s}'.format(kwargs['units']), file=fin)
    else:
       units = 'real'
       print('units     real', file=fin)
    if 'atom_style' in kwargs:
       print('atom_style     {:s}'.format(kwargs['atom_style']), file=fin)
    else:
       print('atom_style     charge', file=fin)
    print('  ', file=fin)
    
    if data != None and data != 'None':
       print('read_data    {:s}'.format(data), file=fin)
       if 'masses' in kwargs:
          # print('masses  ', file=fin)
          for i,s in enumerate(species):
              print('mass',i+1,kwargs['masses'][s], file=fin)
          print('  ', file=fin)
       if 'T' in kwargs:
          print('velocity     all create {:d} {:d}'.format(kwargs['T'],random.randint(0,10000)), file=fin)
       else:
          print('velocity     all create 300 {:d}'.format(random.randint(0,10000)), file=fin)
    if restart != None and restart != 'None':
       print('read_restart {:s}'.format(restart), file=fin)
    print(' ', file=fin)

    print('pair_style     {:s}'.format(pair_style), file=fin) 
    if isinstance(pair_coeff, list):
       for pc in pair_coeff:
           print('pair_coeff     {:s}'.format(pc), file=fin)
    else:
       print('pair_coeff     {:s}'.format(pair_coeff), file=fin)
    if pair_style.find('reaxff')>=0:
       print('compute       reax all pair reaxff', file=fin)
       print('variable eb   equal c_reax[1]', file=fin)
       print('variable ea   equal c_reax[2]', file=fin)
       print('variable elp  equal c_reax[3]', file=fin)
       print('variable emol equal c_reax[4]', file=fin)
       print('variable ev   equal c_reax[5]', file=fin)
       print('variable epen equal c_reax[6]', file=fin)
       print('variable ecoa equal c_reax[7]', file=fin)
       print('variable ehb  equal c_reax[8]', file=fin)
       print('variable et   equal c_reax[9]', file=fin)
       print('variable eco  equal c_reax[10]', file=fin)
       print('variable ew   equal c_reax[11]', file=fin)
       print('variable ep   equal c_reax[12]', file=fin)
       print('variable efi  equal c_reax[13]', file=fin)
       print('variable eqeq equal c_reax[14]', file=fin)
    print(' ', file=fin)
    print('neighbor 2.5  bin', file=fin)
    print('neigh_modify  every 1 delay 1 check no page 200000', file=fin)
    print(' ', file=fin)
    if freeatoms:
       # fixatom = [i for i in range(natoms) if i not in free]
       # print(freeatoms)
       print('group  free id ',end=' ', file=fin)
       for j in freeatoms:
           print(j,end=' ', file=fin) 
       print(' ', file=fin)
       print('group  fixed subtract all free', file=fin)
       print('fix    freeze fixed setforce 0.0 0.0 0.0', file=fin)
       print(' ', file=fin)
       fix = fix.replace('all','free')
    print(fix, file=fin)
    print(fix_modify, file=fin)
    if pair_style.find('reaxff')>=0:
       if freeatoms:
          print('fix    rex free qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff', file=fin)
       else:
          print('fix    rex all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff', file=fin)
       # print('fix    sp  all reaxff/species 1 20 20  species.out', file=fin) # every 1 compute bond-order, per 20 av bo, and per 20 calc species
    print(' ', file=fin)
    print(more_commond, file=fin)
    
    if 'minimize' in kwargs:
        print('min_style   cg', file=fin)
        print('min_modify  line quadratic', file=fin)
        print('minimize    {:s}'.format(kwargs['minimize']), file=fin)

    print(' ', file=fin)
    print('thermo        {:d}'.format(dump_interval), file=fin)
    print(thermo_style, file=fin)
    print(' ', file=fin)
    # timestep = convert(timestep, "time", "ASE", units)
    if units == 'metal':
       timestep = timestep*0.001
    print('timestep      {:f}'.format(timestep), file=fin)
    print(' ', file=fin)
    if 'dump' in kwargs:
       print('dump dump_all {:s}'.format(kwargs['dump']), file=fin) 
    else:
       if pair_style.find('reaxff')>=0:
          print('dump   1 all custom {:d} lammps.trj id type x y z q fx fy fz'.format(dump_interval), file=fin)
       else:
          print('dump   1 all custom {:d} lammps.trj id type x y z fx fy fz'.format(dump_interval), file=fin) 
    print(' ', file=fin)
    
    print(' ', file=fin)
    if restart:
       print('restart       10000 restart', file=fin)
    print('run           %d'  %total, file=fin)
    print(' ', file=fin)
    if restartfile is not None:
       print('write_restart {:s}'.format(restartfile), file=fin)
    print('log           %s'  %log, file=fin)
    print('print \"__end_of_ase_invoked_calculation__\" ', file=fin)
    fin.close()
