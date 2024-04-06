import json as js
import ctypes

import os
import shlex
import shutil
import warnings
from re import IGNORECASE
from re import compile as re_compile
import subprocess
from tempfile import NamedTemporaryFile, mkdtemp
from tempfile import mktemp as uns_mktemp
from threading import Thread
from typing import Any, Dict
import numpy as np

from numpy.linalg import norm

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lammps import (CALCULATION_END_MARK, Prism, convert,
                                    write_lammps_in)
from ase.data import atomic_masses, chemical_symbols
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump

# from .qeq import qeq
from .RadiusCutOff import setRcut
from .reaxfflib import read_ffield,write_lib,write_ffield
from .intCheck import init_bonds
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


class IRFF(Calculator):
    '''Intelligent Machine-Learning ASE calculator based on LAMMPS Python module
       Modified from ASE LAMMPSlib.py
    '''
    name = "lammpsrun"
    implemented_properties = ["energy", "free_energy", "forces", "stress",
                              "energies"]

    # parameters to choose options in LAMMPSRUN
    ase_parameters: Dict[str, Any] = dict(
        specorder=None,
        atorder=True,
        always_triclinic=False,
        reduce_cell=False,
        keep_alive=True,
        keep_tmp_files=False,
        no_data_file=False,
        tmp_dir=None,
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
        pair_style="lj/cut 2.5",
        pair_coeff=["* * 1 1"],
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
        super().__init__(label=label, **kwargs)

        self.prism = None
        self.calls = 0
        self.forces = None
        # thermo_content contains data "written by" thermo_style.
        # It is a list of dictionaries, each dict (one for each line
        # printed by thermo_style) contains a mapping between each
        # custom_thermo_args-argument and the corresponding
        # value as printed by lammps. thermo_content will be
        # re-populated by the read_log method.
        self.thermo_content = []

        if self.parameters['tmp_dir'] is not None:                   ## del
            # If tmp_dir is pointing somewhere, don't remove stuff!
            self.parameters['keep_tmp_files'] = True
        self._lmp_handle = None  # To handle the lmp process

        if self.parameters['tmp_dir'] is None:
            self.parameters['tmp_dir'] = mkdtemp(prefix="LAMMPS-")
        else:
            self.parameters['tmp_dir'] = os.path.realpath(
                self.parameters['tmp_dir'])
            if not os.path.isdir(self.parameters['tmp_dir']):
                os.mkdir(self.parameters['tmp_dir'], 0o755)

        for f in self.parameters['files']:
            shutil.copy(
                f, os.path.join(self.parameters['tmp_dir'],
                                os.path.basename(f)))              ## del

        self.active_learning = active_learning

        if libfile.endswith('.json'):
            lf                  = open(libfile,'r')
            j                   = js.load(lf)
            self.p              = j['p']
            m                   = j['m']
            self.MolEnergy_     = j['MolEnergy']
            self.messages       = j['messages']
            self.BOFunction     = j['BOFunction']
            self.EnergyFunction = j['EnergyFunction']
            self.MessageFunction= j['MessageFunction']
            self.VdwFunction    = j['VdwFunction']
            self.bo_layer       = j['bo_layer']
            self.mf_layer       = j['mf_layer']
            self.be_layer       = j['be_layer']
            self.vdw_layer      = j['vdw_layer']
            if not self.vdw_layer is None:
                self.vdwnn       = True
            else:
                self.vdwnn       = False
            rcut                = j['rcut']
            rcuta               = j['rcutBond']
            re                  = j['rEquilibrium']
            lf.close()
            #self.init_bonds()
            if mol is None:
                self.emol = 0.0
            else:
                mol_ = mol.split('-')[0]
                if mol_ in self.MolEnergy_:
                   self.emol = self.MolEnergy_[mol_]
                else:
                   self.emol = 0.0

            self.spec,self.bonds,self.offd,self.angs,self.torp,self.Hbs = init_bonds(self.p)
            write_ffield(self.p,self.spec,self.bonds,self.offd,self.angs,self.torp,self.Hbs,
                        m=m,mf_layer=self.mf_layer,be_layer=self.be_layer,
                        libfile='ffield')
        else:
            self.p,zpe_,self.spec,self.bonds,self.offd,self.angs,self.torp,self.Hbs= \
                        read_ffield(libfile=libfile,zpe=False)
            m             = None
            self.bo_layer = None
            self.emol     = 0.0
            rcut          = None
            rcuta         = None
            re            = None
            self.vdwnn    = False
            self.EnergyFunction  = 0
            self.MessageFunction = 0
            self.VdwFunction     = 0
            self.p['acut']   = 0.0001
            self.p['hbtol']  = 0.0001
        # self.atoms      = atoms
        self.cell         = atoms.get_cell()
        self.atom_name    = atoms.get_chemical_symbols()
        self.natom        = len(self.atom_name)
        
        self.nn        = False if m is None else True
        self.hbshort   = hbshort
        self.hblong    = hblong
        self.set_rcut(rcut,rcuta,re)
        self.vdwcut    = vdwcut
        self.botol     = 0.01*self.p['cutoff']
        self.atol      = self.p['acut']   # atol
        self.hbtol     = self.p['hbtol']  # hbtol
        self.check_offd()
        self.check_hb()
        self.get_rcbo()
        self.set_p(m,self.bo_layer)
        # self.Qe= qeq(p=self.p,atoms=self.atoms)

    def get_lammps_command(self):
        cmd = self.parameters.get('command')

        if cmd is None:
            from ase.config import cfg
            envvar = f'ASE_{self.name.upper()}_COMMAND'
            cmd = cfg.get(envvar)

        if cmd is None:
           # TODO deprecate and remove guesswork
           cmd = 'lammps'

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
        lammps_in = uns_mktemp(
            prefix="in_" + label, dir=tempdir
        )
        lammps_log = uns_mktemp(
            prefix="log_" + label, dir=tempdir
        )
        lammps_trj_fd = NamedTemporaryFile(
            prefix="trj_" + label,
            suffix=(".bin" if self.parameters['binary_dump'] else ""),
            dir=tempdir,
            delete=(not self.parameters['keep_tmp_files']),
        )
        lammps_trj = lammps_trj_fd.name
        if self.parameters['no_data_file']:
            lammps_data = None
        else:
            lammps_data_fd = NamedTemporaryFile(
                prefix="data_" + label,
                dir=tempdir,
                delete=(not self.parameters['keep_tmp_files']),
                mode='w',
                encoding='ascii'
            )
            write_lammps_data(
                lammps_data_fd,
                self.atoms,
                specorder=self.parameters['specorder'],
                force_skew=self.parameters['always_triclinic'],
                # reduce_cell=self.parameters['reduce_cell'],
                velocities=self.parameters['write_velocities'],
                prismobj=self.prism,
                units=self.parameters['units'],
                atom_style=self.parameters['atom_style'],
            )
            lammps_data = lammps_data_fd.name
            lammps_data_fd.flush()

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
        if self.parameters['keep_tmp_files']:
            lammps_log_fd = open(lammps_log, "w")
            fd = SpecialTee(lmp_handle.stdout, lammps_log_fd)
        else:
            fd = lmp_handle.stdout
        thr_read_log = Thread(target=self.read_lammps_log, args=(fd,))
        thr_read_log.start()

        # write LAMMPS input (for reference, also create the file lammps_in,
        # although it is never used)
        if self.parameters['keep_tmp_files']:
            lammps_in_fd = open(lammps_in, "w")
            fd = SpecialTee(lmp_handle.stdin, lammps_in_fd)
        else:
            fd = lmp_handle.stdin
        write_lammps_in(
            lammps_in=fd,
            parameters=self.parameters,
            atoms=self.atoms,
            prismobj=self.prism,
            lammps_trj=lammps_trj,
            lammps_data=lammps_data,
        )

        if self.parameters['keep_tmp_files']:
            lammps_in_fd.close()

        # Wait for log output to be read (i.e., for LAMMPS to finish)
        # and close the log file if there is one
        thr_read_log.join()
        if self.parameters['keep_tmp_files']:
            lammps_log_fd.close()

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
        stress = np.array(
            [-tc[i] for i in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")]
        )

        # We need to apply the Lammps rotation stuff to the stress:
        xx, yy, zz, yz, xz, xy = stress
        stress_tensor = np.array([[xx, xy, xz],
                                  [xy, yy, yz],
                                  [xz, yz, zz]])
        stress_atoms = self.prism.tensor2_to_ase(stress_tensor)
        stress_atoms = stress_atoms[[0, 1, 2, 1, 0, 0],
                                    [0, 1, 2, 2, 2, 1]]
        stress = stress_atoms

        self.results["stress"] = convert(
            stress, "pressure", self.parameters["units"], "ASE"
        )

        lammps_trj_fd.close()
        if not self.parameters['no_data_file']:
            lammps_data_fd.close()

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

    def get_bond_energy(self, atoms, properties=['energy', 'forces','charges','stress'],
                  system_changes=['positions',  'cell','numbers', 'charges']):
        if self.active_learning:
           self.calculate_bond_order(atoms)
        self.propagate(atoms, properties, system_changes, 0)

    def check_hb(self):
        if 'H' in self.spec:
            for sp1 in self.spec:
                if sp1 != 'H':
                    for sp2 in self.spec:
                        if sp2 != 'H':
                            hb = sp1+'-H-'+sp2
                            if hb not in self.Hbs:
                                self.Hbs.append(hb) # 'rohb','Dehb','hb1','hb2'
                                self.p['rohb_'+hb] = 1.9
                                self.p['Dehb_'+hb] = 0.0
                                self.p['hb1_'+hb]  = 2.0
                                self.p['hb2_'+hb]  = 19.0

    def check_offd(self):
        p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
        for key in p_offd:
            for sp in self.spec:
                try:
                    self.p[key+'_'+sp+'-'+sp]  = self.p[key+'_'+sp]  
                except KeyError:
                    print('-  warning: key not in dict') 

        for bd in self.bonds:             # check offd parameters
            b= bd.split('-')
            if 'rvdw_'+bd not in self.p:
                for key in p_offd:        # set offd parameters according combine rules
                    if self.p[key+'_'+b[0]]>0.0 and self.p[key+'_'+b[1]]>0.0:
                        self.p[key+'_'+bd] = np.sqrt(self.p[key+'_'+b[0]]*self.p[key+'_'+b[1]])
                    else:
                        self.p[key+'_'+bd] = -1.0

        for bd in self.bonds:             # check minus ropi ropp parameters
            if self.p['ropi_'+bd]<0.0:
                self.p['ropi_'+bd] = 0.3*self.p['rosi_'+bd]
                self.p['bo3_'+bd]  = -50.0
                self.p['bo4_'+bd]  = 0.0
            if self.p['ropp_'+bd]<0.0:
                self.p['ropp_'+bd] = 0.2*self.p['rosi_'+bd]
                self.p['bo5_'+bd]  = -50.0
                self.p['bo6_'+bd]  = 0.0

    def set_rcut(self,rcut,rcuta,re): 
        rcut_,rcuta_,re_ = setRcut(self.bonds,rcut,rcuta,re)
        self.rcut  = rcut_    ## bond order compute cutoff
        self.rcuta = rcuta_  ## angle term cutoff

        self.r_cut = np.zeros([self.natom,self.natom],dtype=np.float32)
        self.r_cuta = np.zeros([self.natom,self.natom],dtype=np.float32)
        self.re = np.zeros([self.natom,self.natom],dtype=np.float32)
        for i in range(self.natom):
            for j in range(self.natom):
                bd = self.atom_name[i] + '-' + self.atom_name[j]
                if i!=j:
                    self.r_cut[i][j]  = self.rcut[bd]  
                    self.r_cuta[i][j] = self.rcuta[bd] 
                    self.re[i][j]     = re_[bd] 
                    # print(i,j,bd,re_[bd])

    def get_rcbo(self):
        ''' get cut-offs for individual bond '''
        self.rc_bo = {}
        for bd in self.bonds:
            b= bd.split('-')
            ofd=bd if b[0]!=b[1] else b[0]

            log_ = np.log((self.botol/(1.0+self.botol)))
            # rr = log_/self.p['bo1_'+bd] 
            self.rc_bo[bd]=self.p['rosi_'+ofd]*np.power(log_/self.p['bo1_'+bd],1.0/self.p['bo2_'+bd])
    
    def get_bondorder_uc(self):
        if self.nn:
            self.frc = 1.0
        else:
            self.frc = np.where(np.logical_or(self.r>self.rcbo,self.r<=0.0001), 0.0,1.0)

        self.bodiv1 = self.r/self.P['rosi']
        self.bopow1 = np.power(self.bodiv1,self.P['bo2'])
        self.eterm1 = (1.0+self.botol)*np.exp(self.P['bo1']*self.bopow1)*self.frc # consist with GULP

        self.bodiv2 = self.r/self.P['ropi']
        self.bopow2 = np.power(self.bodiv2,self.P['bo4'])
        self.eterm2 = np.exp(self.P['bo3']*self.bopow2)*self.frc

        self.bodiv3 = self.r/self.P['ropp']
        self.bopow3 = np.power(self.bodiv3,self.P['bo6'])
        self.eterm3 = np.exp(self.P['bo5']*self.bopow3)*self.frc

        if self.nn:
            if self.BOFunction==0:
                fsi_  = taper(self.eterm1,rmin=self.botol,rmax=2.0*self.botol)*(self.eterm1-self.botol)
                fpi_  = taper(self.eterm2,rmin=self.botol,rmax=2.0*self.botol)*self.eterm2
                fpp_  = taper(self.eterm3,rmin=self.botol,rmax=2.0*self.botol)*self.eterm3
            elif self.BOFunction==1:
                fsi_  = self.f_nn('fsi',[self.eterm1],layer=self.bo_layer[1])
                fpi_  = self.f_nn('fpi',[self.eterm2],layer=self.bo_layer[1])
                fpp_  = self.f_nn('fpp',[self.eterm3],layer=self.bo_layer[1])  
            elif self.BOFunction==2:
                fsi_  = self.f_nn('fsi',[-self.eterm1],layer=self.bo_layer[1])
                fpi_  = self.f_nn('fpi',[-self.eterm2],layer=self.bo_layer[1])
                fpp_  = self.f_nn('fpp',[-self.eterm3],layer=self.bo_layer[1])
            else:
                raise NotImplementedError('-  BO function not supported yet!')
            self.bop_si = fsi_*self.eye #*self.frc #*self.eterm1
            self.bop_pi = fpi_*self.eye #*self.frc #*self.eterm2
            self.bop_pp = fpp_*self.eye #*self.frc #*self.eterm3
        else:
            self.bop_si = taper(self.eterm1,rmin=self.botol,rmax=2.0*self.botol)*(self.eterm1-self.botol) # consist with GULP
            self.bop_pi = taper(self.eterm2,rmin=self.botol,rmax=2.0*self.botol)*self.eterm2
            self.bop_pp = taper(self.eterm3,rmin=self.botol,rmax=2.0*self.botol)*self.eterm3
        self.bop    = self.bop_si+self.bop_pi+self.bop_pp

        self.Deltap= np.sum(self.bop,axis=1)  
        if self.MessageFunction==1:
            self.D_si = [np.sum(self.bop_si,axis=1)] 
            self.D_pi = [np.sum(self.bop_pi,axis=1)] 
            self.D_pp = [np.sum(self.bop_pp,axis=1)] 

    def f_nn(self,pre,x,layer=5):
        X   = np.expand_dims(np.stack(x,axis=2),2)

        o   = []
        o.append(sigmoid(np.matmul(X,self.m[pre+'wi'])+self.m[pre+'bi']))  
                                                                        # input layer
        for l in range(layer):                                        # hidden layer      
            o.append(sigmoid(np.matmul(o[-1],self.m[pre+'w'][l])+self.m[pre+'b'][l]))
        
        o_  = sigmoid(np.matmul(o[-1],self.m[pre+'wo']) + self.m[pre+'bo']) 
        out = np.squeeze(o_)                                          # output layer
        return out

    def message_passing(self):
        self.H         = []    # hiden states (or embeding states)
        self.D         = []    # degree matrix
        self.Hsi       = []
        self.Hpi       = []
        self.Hpp       = []
        self.F         = []
        self.H.append(self.bop)                   # 
        self.Hsi.append(self.bop_si)              #
        self.Hpi.append(self.bop_pi)              #
        self.Hpp.append(self.bop_pp)              # 
        self.D.append(self.Deltap)                # get the initial hidden state H[0]

        for t in range(1,self.messages+1):
            Di        = np.expand_dims(self.D[t-1],axis=0)*self.eye
            Dj        = np.expand_dims(self.D[t-1],axis=1)*self.eye

            if self.MessageFunction==1:
                Dsi_i = np.expand_dims(self.D_si[t-1],axis=0)*self.eye - self.Hsi[t-1]
                Dsi_j = np.expand_dims(self.D_si[t-1],axis=1)*self.eye - self.Hsi[t-1]

                Dpi_i = np.expand_dims(self.D_pi[t-1],axis=0)*self.eye - self.Hpi[t-1]
                Dpi_j = np.expand_dims(self.D_pi[t-1],axis=1)*self.eye - self.Hpi[t-1]

                Dpp_i = np.expand_dims(self.D_pp[t-1],axis=0)*self.eye - self.Hpp[t-1]
                Dpp_j = np.expand_dims(self.D_pp[t-1],axis=1)*self.eye - self.Hpp[t-1]

                Dpii  = Dpi_i + Dpp_i
                Dpij  = Dpi_j + Dpp_j

                # Fi  = self.f_nn('f'+str(t),[Dsi_j,Dpi_j,Dpp_j,Dsi_i,Dpi_i,Dpp_i,self.Hsi[t-1],self.Hpi[t-1],self.Hpp[t-1]],
                #                  layer=self.mf_layer[1])
                Fi  = self.f_nn('fm',[Dsi_i,Dpii,self.H[t-1],Dsi_j,Dpij],layer=self.mf_layer[1])
                Fj  = np.transpose(Fi,[1,0,2])
                F   = Fi*Fj
                Fsi = F[:,:,0]
                Fpi = F[:,:,1]
                Fpp = F[:,:,2]
                self.F.append(F)
                self.Hsi.append(self.Hsi[t-1]*Fsi)
                self.Hpi.append(self.Hpi[t-1]*Fpi)
                self.Hpp.append(self.Hpp[t-1]*Fpp)
            elif self.MessageFunction==2:
                Dbi   = Di  - self.H[t-1]
                Dbj   = Dj  - self.H[t-1]
                #Fi    = self.f_nn('fm',[Dbj,Dbi,self.Hsi[t-1],self.Hpi[t-1],self.Hpp[t-1]], # +str(t)
                #                  layer=self.mf_layer[1])
                Fi    = self.f_nn('fm',[Dbj,self.H[t-1],Dbi],layer=self.mf_layer[1])
                Fj    = np.transpose(Fi,[1,0,2])
                F     = Fi*Fj
                Fsi = F[:,:,0]
                Fpi = F[:,:,1]
                Fpp = F[:,:,2]
                self.F.append(F)
                self.Hsi.append(self.Hsi[t-1]*Fsi)
                self.Hpi.append(self.Hpi[t-1]*Fpi)
                self.Hpp.append(self.Hpp[t-1]*Fpp)
            elif self.MessageFunction==3:
                Dbi   = Di - self.H[t-1] # np.expand_dims(self.P['valboc'],axis=0) 
                Dbj   = Dj - self.H[t-1] # np.expand_dims(self.P['valboc'],axis=1)  
                Fi    = self.f_nn('fm',[Dbj,self.H[t-1],Dbi],layer=self.mf_layer[1]) # +str(t)
                #Fj   = self.f_nn('f'+str(t),[Dbi,self.H[t-1],Dbj],layer=self.mf_layer[1])
                Fj    = np.transpose(Fi,[1,0,2])
                F     = Fi*Fj
                self.F.append(F)
                Fsi   = F[:,:,0]
                Fpi   = F[:,:,1]
                Fpp   = F[:,:,2]
                self.Hsi.append(self.Hsi[t-1]*Fsi)
                self.Hpi.append(self.Hpi[t-1]*Fpi)
                self.Hpp.append(self.Hpp[t-1]*Fpp)
            elif self.MessageFunction==4:
                Dbi   = Di - np.expand_dims(self.P['val'],axis=0) 
                Dbj   = Dj - np.expand_dims(self.P['val'],axis=1)
                Fi    = self.f_nn('fm',[Dbj,self.H[t-1],Dbi],layer=self.mf_layer[1])   # +str(t)
                Fj    = self.f_nn('fm',[Dbi,self.H[t-1],Dbj],layer=self.mf_layer[1])   # +str(t)
                
                #  self.f1()
                #  f11 = self.f_1*self.f_1
                #  F11 = np.where(self.P['ovcorr']>=0.0001,f11,1.0)

                F     = Fi*Fj #*F11     # By default p_corr13 is always True
                self.F.append(F)
                self.Hsi.append(self.Hsi[t-1]*F)
                self.Hpi.append(self.Hpi[t-1]*F)
                self.Hpp.append(self.Hpp[t-1]*F)
            elif self.MessageFunction==5:
                Dbi   = Di - np.expand_dims(self.P['val'],axis=0) # Di  - self.H[t-1]
                Dbj   = Dj - np.expand_dims(self.P['val'],axis=1) # Dj  - self.H[t-1]
                Fi    = self.f_nn('fm',[Dbj,self.H[t-1],Dbi],layer=self.mf_layer[1]) # +str(t)
                Fj    = self.f_nn('fm',[Dbi,self.H[t-1],Dbj],layer=self.mf_layer[1]) # +str(t)
                #Fj    = np.transpose(Fi,[2,1,0])
                F     = Fi*Fj
                self.F.append(F)
                Fsi   = F[:,:,0]
                Fpi   = F[:,:,1]
                Fpp   = F[:,:,2]
                self.Hsi.append(Fsi)
                self.Hpi.append(Fpi)
                self.Hpp.append(Fpp)
            else:
                raise NotImplementedError('-  Message function not supported yet!')
            self.H.append(self.Hsi[t]+self.Hpi[t]+self.Hpp[t])
            self.D.append(np.sum(self.H[t],axis=1))  
            if self.MessageFunction==1:
                self.D_si.append(np.sum(self.Hsi[t],axis=1))
                self.D_pi.append(np.sum(self.Hpi[t],axis=1))
                self.D_pp.append(np.sum(self.Hpp[t],axis=1))

    def get_bondorder_nn(self):
        self.message_passing()
        self.bosi  = self.Hsi[-1]       # getting the final state
        self.bopi  = self.Hpi[-1]
        self.bopp  = self.Hpp[-1]

        self.bo0   = self.H[-1] # np.where(self.H[-1]<0.0000001,0.0,self.H[-1]) # self.bosi + self.bopi + self.bopp

        # self.fbo   = taper(self.bo0,rmin=self.botol,rmax=2.0*self.botol)
        self.bo    = relu(self.bo0 - self.atol*self.eye)     # bond-order cut-off 0.001 reaxffatol
        self.bso   = self.P['ovun1']*self.P['Desi']*self.bo0  
        self.Delta = np.sum(self.bo0,axis=1)   

        self.Di    = np.expand_dims(self.Delta,axis=0)*self.eye          # get energy layer
        self.Dj    = np.expand_dims(self.Delta,axis=1)*self.eye
        Dbi        = self.Di - self.bo0
        Dbj        = self.Dj - self.bo0

        if self.EnergyFunction==0:
            self.powb  = np.power(self.bosi+self.safety_value,self.P['be2'])
            self.expb  = np.exp(np.multiply(self.P['be1'],1.0-self.powb))
            self.sieng = self.P['Desi']*self.bosi*self.expb 

            self.pieng = np.multiply(self.P['Depi'],self.bopi)
            self.ppeng = np.multiply(self.P['Depp'],self.bopp)
            self.esi   = self.sieng + self.pieng + self.ppeng
        elif self.EnergyFunction==1: 
            esi      = self.f_nn('fe',[self.bosi,self.bopi,self.bopp],layer=self.be_layer[1])
            self.esi = esi*np.where(self.bo0<0.0000001,0.0,1.0)
        elif self.EnergyFunction==2:
            self.esi = self.f_nn('fe',[-self.bosi,-self.bopi,-self.bopp],layer=self.be_layer[1]) 
            self.esi = self.esi*np.where(self.bo0<0.0000001,0.0,1.0)
        elif self.EnergyFunction==3: 
            e_ = self.f_nn('fe',[self.bosi,self.bopi,self.bopp],layer=self.be_layer[1])  
            self.esi = self.bo0*e_
        elif self.EnergyFunction==4:
            Fi = self.f_nn('fe',[Dbj,Dbi,self.bo0],layer=self.be_layer[1])
            Fj = np.transpose(Fi,[1,0])
            self.esi = Fi*Fj*self.bo0
        # elif self.EnergyFunction==5:
        #    r_        = self.bodiv1
        #    mors_exp1 = np.exp(self.P['be2']*(1.0-r_))
        #    mors_exp2 = np.square(mors_exp1) 
        #    mors_exp10= np.exp(self.P['be2']*self.P['be1'])
        #    mors_exp20= np.square(mors_exp10) 
        #    emorse    = 2.0*mors_exp1 - mors_exp2 + mors_exp20 - 2.0*mors_exp10
        #    self.esi  = np.maximum(0,emorse)
        elif self.EnergyFunction==5:
            self.sieng = np.multiply(self.P['Desi'],self.bosi)
            self.pieng = np.multiply(self.P['Depi'],self.bopi)
            self.ppeng = np.multiply(self.P['Depp'],self.bopp)
            self.esi   = self.sieng + self.pieng - self.ppeng
        else:
            raise NotImplementedError('-  This method is not implimented!')

    def get_ebond(self,cell,rcell,positions):
        self.vr    = fvr(positions)
        vrf        = np.dot(self.vr,rcell)

        vrf        = np.where(vrf-0.5>0,vrf-1.0,vrf)
        vrf        = np.where(vrf+0.5<0,vrf+1.0,vrf) 

        self.vr    = np.dot(vrf,cell)
        self.r     = np.sqrt(np.sum(self.vr*self.vr,axis=2)+self.safety_value)

        self.get_bondorder_uc()

        if self.nn:
            self.get_bondorder_nn()
        else:
            self.get_bondorder()

        self.Dv    = self.Delta - self.P['val']
        self.Dpi   = np.sum(self.bopi+self.bopp,axis=1) 

        self.so    = np.sum(self.P['ovun1']*self.P['Desi']*self.bo0,axis=1)  
        self.fbo   = taper(self.bo0,rmin=self.atol,rmax=2.0*self.atol) 
        self.fhb   = taper(self.bo0,rmin=self.hbtol,rmax=2.0*self.hbtol) 

        if self.EnergyFunction>=1: # or self.EnergyFunction==2 or self.EnergyFunction==3:
            self.ebond = - self.P['Desi']*self.esi
        else:
            #if self.nn:
            self.ebond = - self.esi
            #else:
        self.Ebond = 0.5*np.sum(self.ebond)
        return self.Ebond
        
    def calculate_bond_order(self,atoms=None):
        cell      = atoms.get_cell()                    # cell is object now
        cell      = cell[:].astype(dtype=np.float32)
        rcell     = np.linalg.inv(cell).astype(dtype=np.float32)

        positions = atoms.get_positions()
        xf        = np.dot(positions,rcell)
        xf        = np.mod(xf,1.0)
        positions = np.dot(xf,cell).astype(dtype=np.float32)
        self.get_ebond(cell,rcell,positions)
            
    def set_p(self,m,bo_layer):
        ''' setting up parameters '''
        self.unit   = 4.3364432032e-2
        self.punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
                        'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb']
        p_bond = ['Desi','Depi','Depp','bo5','bo6','ovun1',
                    'be1','be2','bo3','bo4','bo1','bo2',
                    'Devdw','rvdw','alfa','rosi','ropi','ropp',
                    'corr13','ovcorr']
        p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
        self.P = {}
        
        if not self.nn:
            self.p['lp3'] = 75.0
        # else:
        #    self.hbtol = self.p['hbtol']
        #    self.atol = self.p['acut']  
        self.rcbo = np.zeros([self.natom,self.natom],dtype=np.float32)

        for i in range(self.natom):
            for j in range(self.natom):
                bd = self.atom_name[i] + '-' + self.atom_name[j]
                if bd not in self.bonds:
                    bd = self.atom_name[j] + '-' + self.atom_name[i]
                self.rcbo[i][j] = min(self.rcut[bd],self.rc_bo[bd])   #  ###### TODO #####

        p_spec = ['valang','valboc','val','vale',
                    'lp2','ovun5',                 # 'val3','val5','boc3','boc4','boc5'
                    'ovun2','atomic',
                    'mass','chi','mu']             # 'gamma','gammaw','Devdw','rvdw','alfa'

        for key in p_spec:
            unit_ = self.unit if key in self.punit else 1.0
            self.P[key] = np.zeros([self.natom],dtype=np.float32)
            for i in range(self.natom):
                    sp = self.atom_name[i]
                    self.P[key][i] = self.p[key+'_'+sp]*unit_
        self.zpe = -np.sum(self.P['atomic']) + self.emol

        for key in ['boc3','boc4','boc5','gamma','gammaw']:
            self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float32)
            for i in range(self.natom):
                for j in range(self.natom):
                    self.P[key][i][j] = np.sqrt(self.p[key+'_'+self.atom_name[i]]*self.p[key+'_'+self.atom_name[j]],
                                                dtype=np.float32)
        
        for key in p_bond:
            unit_ = self.unit if key in self.punit else 1.0
            self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float32)
            for i in range(self.natom):
                for j in range(self.natom):
                    bd = self.atom_name[i] + '-' + self.atom_name[j]
                    if bd not in self.bonds:
                        bd = self.atom_name[j] + '-' + self.atom_name[i]
                    self.P[key][i][j] = self.p[key+'_'+bd]*unit_
        
        p_g  = ['boc1','boc2','coa2','ovun6','lp1','lp3',
                'ovun7','ovun8','val6','val9','val10','tor2',
                'tor3','tor4','cot2','coa4','ovun4',               
                'ovun3','val8','coa3','pen2','pen3','pen4','vdw1'] 
        for key in p_g:
            self.P[key] = self.p[key]
    
        # for key in self.p_ang:
        #     unit_ = self.unit if key in self.punit else 1.0
        #     for a in self.Angs:
        #         pn = key + '_' + a
        #         self.p[pn] = self.p[pn]*unit_

        # for key in self.p_tor:
        #     unit_ = self.unit if key in self.punit else 1.0
        #     for t in self.Tors:
        #         pn = key + '_' + t
        #         if pn in self.p:
        #             self.p[pn] = self.p[pn]*unit_
        #         else:
        #             self.p[pn] = 0.0

        # for h in self.Hbs:
        #     pn = 'Dehb_' + h
        #     self.p[pn] = self.p[pn]*self.unit

        if self.nn:
            self.set_m(m)


    def set_m(self,m):
        self.m = {}
        if self.BOFunction==0:
            pres = ['fe','fm']
        else:
            pres = ['fe','fsi','fpi','fpp','fm']
            
        if self.vdwnn:
            pres.append('fv')

        # for t in range(1,self.messages+1):
        #     pres.append('fm') # +str(t)

        for k_ in pres:
            for k in ['wi','bi','wo','bo']:
                key = k_+k
                self.m[key] = []
                for i in range(self.natom):
                    mi_ = []
                    for j in range(self.natom):
                        if k_ in ['fe','fsi','fpi','fpp','fv']:
                            bd = self.atom_name[i] + '-' + self.atom_name[j]
                            if bd not in self.bonds:
                                bd = self.atom_name[j] + '-' + self.atom_name[i]
                        else:
                            bd = self.atom_name[i] 
                        key_ = key+'_'+bd
                        if key_ in m:
                            if k in ['bi','bo']:
                                mi_.append(np.expand_dims(m[key+'_'+bd],axis=0))
                            else:
                                mi_.append(m[key+'_'+bd])
                    self.m[key].append(mi_)
                self.m[key] = np.array(self.m[key],dtype=np.float32)

            for k in ['w','b']:
                key = k_+k
                self.m[key] = []

                if k_ in ['fesi','fepi','fepp','fe']:
                    layer_ = self.be_layer[1]
                elif k_ in ['fsi','fpi','fpp']:
                    layer_ = self.bo_layer[1]
                elif k_ =='fv':
                    layer_ = self.vdw_layer[1]
                else:
                    layer_ = self.mf_layer[1]

                for l in range(layer_):
                    m_ = []
                    for i in range(self.natom):
                        mi_ = []
                        for j in range(self.natom):
                            if k_ in ['fe','fsi','fpi','fpp','fv']:
                                bd = self.atom_name[i] + '-' + self.atom_name[j]
                                if bd not in self.bonds:
                                    bd = self.atom_name[j] + '-' + self.atom_name[i]
                            else:
                                bd = self.atom_name[i] 
                            key_ = key+'_'+bd
                            if key_ in m:
                                if k == 'b':
                                    mi_.append(np.expand_dims(m[key+'_'+bd][l],axis=0))
                                else:
                                    mi_.append(m[key+'_'+bd][l])
                        m_.append(mi_)
                    self.m[key].append(np.array(m_,dtype=np.float32))

    def set_m_univeral(self,m,pres=['fe','fsi','fpi','fpp']):
        # pres = ['fe','fsi','fpi','fpp']   # general neural network matrix
        for t in range(1,self.messages+1):
            pres.append('fm') # +str(t)

        for k_ in pres:
            for k in ['wi','bi','wo','bo']:
                key = k_+k
                if k in ['bi','bo']:
                    mi_ = np.expand_dims(np.expand_dims(np.expand_dims(m[key],axis=0),axis=0),axis=0)
                else:
                    mi_ = np.expand_dims(np.expand_dims(m[key],axis=0),axis=0)
                self.m[key] = np.array(mi_,dtype=np.float32)

            for k in ['w','b']:
                key = k_+k
                self.m[key] = []

                if k_ in ['fesi','fepi','fepp','fe']:
                    layer_ = self.be_layer[1]
                elif k_ in ['fsi','fpi','fpp']:
                    layer_ = self.bo_layer[1]
                elif k_ =='fv':
                    layer_ = self.vdw_layer[1]
                else:
                    layer_ = self.mf_layer[1]

                for l in range(layer_):
                    if k == 'b':
                        m_ = np.expand_dims(np.expand_dims(np.expand_dims(m[key][l],axis=0),axis=0),axis=0)
                    else:
                        m_ = np.expand_dims(np.expand_dims(m[key][l],axis=0),axis=0)
                    self.m[key].append(np.array(m_,dtype=np.float32))

 

class SpecialTee:
    """A special purpose, with limited applicability, tee-like thing.

    A subset of stuff read from, or written to, orig_fd,
    is also written to out_fd.
    It is used by the lammps calculator for creating file-logs of stuff
    read from, or written to, stdin and stdout, respectively.
    """

    def __init__(self, orig_fd, out_fd):
        self._orig_fd = orig_fd
        self._out_fd = out_fd
        self.name = orig_fd.name

    def write(self, data):
        self._orig_fd.write(data)
        self._out_fd.write(data)
        self.flush()

    def read(self, *args, **kwargs):
        data = self._orig_fd.read(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readline(self, *args, **kwargs):
        data = self._orig_fd.readline(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readlines(self, *args, **kwargs):
        data = self._orig_fd.readlines(*args, **kwargs)
        self._out_fd.write("".join(data))
        return data

    def flush(self):
        self._orig_fd.flush()
        self._out_fd.flush()
