import json as js
import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
                      chemical_symbols as ase_chemical_symbols,
                      atomic_masses as ase_atomic_masses)
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
from ase import Atoms
from ase.io import read,write
from ase.calculators.calculator import Calculator, all_changes
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
    name = "IRFF"
    implemented_properties = ['energy', 'forces', 'stress']

    started = False
    initialized = False

    default_parameters = dict(
            atom_types=None,
            atom_type_masses=None,
            log_file=None,
            lammps_name='',
            keep_alive=True,
            lammps_header=['units real',
                           'atom_style charge',
                           'atom_modify map array sort 0 0'],
            amendments=None,
            post_changebox_cmds=None,
            boundary=True,
            create_box=True,
            create_atoms=True,
            read_molecular_info=False,
            comm=None)

    def __init__(self,
                 mol=None,atoms=None,
                 libfile='ffield.json',
                 vdwcut=10.0,
                 hbshort=6.75,hblong=7.5,
                 label="IRFF", 
                 active_learning=False,
                 *args,**kwargs):
        Calculator.__init__(self,label=label, **kwargs)
        self.spec         = []
        self.GPa          = 1.60217662*1.0e2
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
        self.lmp = None

    def set_cell(self, atoms, change=False):
        lammps_cell, self.coord_transform = convert_cell(atoms.get_cell())

        xhi, xy, xz, _, yhi, yz, _, _, zhi = convert(
            lammps_cell.flatten(order='C'), "distance", "ASE", self.units)
        box_hi = [xhi, yhi, zhi]

        if change:
            cell_cmd = ('change_box all     '
                        'x final 0 {} y final 0 {} z final 0 {}      '
                        'xy final {} xz final {} yz final {} units box'
                        ''.format(xhi, yhi, zhi, xy, xz, yz))
            if self.parameters.post_changebox_cmds is not None:
                for cmd in self.parameters.post_changebox_cmds:
                    self.lmp.command(cmd)
        else:
            # just in case we'll want to run with a funny shape box,
            # and here command will only happen once, and before
            # any calculation
            if self.parameters.create_box:
                self.lmp.command('box tilt large')

            # Check if there are any indefinite boundaries. If so,
            # shrink-wrapping will end up being used, but we want to
            # define the LAMMPS region and box fairly tight around the
            # atoms to avoid losing any
            lammps_boundary_conditions = self.lammpsbc(atoms).split()
            if 's' in lammps_boundary_conditions:
                pos = atoms.get_positions()
                if self.coord_transform is not None:
                    pos = np.dot(self.coord_transform, pos.transpose())
                    pos = pos.transpose()
                posmin = np.amin(pos, axis=0)
                posmax = np.amax(pos, axis=0)

                for i in range(0, 3):
                    if lammps_boundary_conditions[i] == 's':
                        box_hi[i] = 1.05 * abs(posmax[i] - posmin[i])

            cell_cmd = ('region cell prism    '
                        '0 {} 0 {} 0 {}     '
                        '{} {} {}     units box'
                        ''.format(*box_hi, xy, xz, yz))

        self.lmp.command(cell_cmd)

    def set_lammps_pos(self, atoms):
        # Create local copy of positions that are wrapped along any periodic
        # directions
        cell = convert(atoms.cell, "distance", "ASE", self.units)
        pos = convert(atoms.positions, "distance", "ASE", self.units)

        # If necessary, transform the positions to new coordinate system
        if self.coord_transform is not None:
            pos = np.dot(pos, self.coord_transform.T)
            cell = np.dot(cell, self.coord_transform.T)

        # wrap only after scaling and rotating to reduce chances of
        # lammps neighbor list bugs.
        pos = wrap_positions(pos, cell, atoms.get_pbc())

        # Convert ase position matrix to lammps-style position array
        # contiguous in memory
        lmp_positions = list(pos.ravel())

        # Convert that lammps-style array into a C object
        c_double_array = (ctypes.c_double * len(lmp_positions))
        lmp_c_positions = c_double_array(*lmp_positions)
        #        self.lmp.put_coosrds(lmp_c_positions)
        self.lmp.scatter_atoms('x', 1, 3, lmp_c_positions)

    def calculate(self, atoms, properties=['energy', 'forces','charges','stress'],
                  system_changes=['positions',  'cell','numbers', 'charges']):
        if self.active_learning:
           self.calculate_bond_order(atoms)
        self.propagate(atoms, properties, system_changes, 0)

    def propagate(self, atoms, properties, system_changes, n_steps, dt=None,
                  dt_not_real_time=False, velocity_field=None):
        """"atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'charges' and 'magmoms'.
        """
        if len(system_changes) == 0:
            return

        self.coord_transform = None

        if not self.started:
            self.start_lammps()
        if not self.initialized:
            self.initialise_lammps(atoms)
        else:  # still need to reset cell
            # NOTE: The whole point of ``post_changebox_cmds`` is that they're
            # executed after any call to LAMMPS' change_box command.  Here, we
            # rely on the fact that self.set_cell(), where we have currently
            # placed the execution of ``post_changebox_cmds``, gets called
            # after this initial change_box call.

            # Apply only requested boundary condition changes.  Note this needs
            # to happen before the call to set_cell since 'change_box' will
            # apply any shrink-wrapping *after* it's updated the cell
            # dimensions
            if 'pbc' in system_changes:
                change_box_str = 'change_box all boundary {}'
                change_box_cmd = change_box_str.format(self.lammpsbc(atoms))
                self.lmp.command(change_box_cmd)

            # Reset positions so that if they are crazy from last
            # propagation, change_box (in set_cell()) won't hang.
            # Could do this only after testing for crazy positions?
            # Could also use scatter_atoms() to set values (requires
            # MPI comm), or extra_atoms() to get pointers to local
            # data structures to zero, but then we would have to be
            # careful with parallelism.
            self.lmp.command("set atom * x 0.0 y 0.0 z 0.0")
            self.set_cell(atoms, change=True)

        if self.parameters.atom_types is None:
            raise NameError("atom_types are mandatory.")

        do_rebuild = (not np.array_equal(atoms.numbers,
                                         self.previous_atoms_numbers)
                      or ("numbers" in system_changes))
        if not do_rebuild:
            do_redo_atom_types = not np.array_equal(
                atoms.numbers, self.previous_atoms_numbers)
        else:
            do_redo_atom_types = False

        self.lmp.command('echo none')  # don't echo the atom positions
        if do_rebuild:
            self.rebuild(atoms)
        elif do_redo_atom_types:
            self.redo_atom_types(atoms)
        self.lmp.command('echo log')  # switch back log

        self.set_lammps_pos(atoms)

        if self.parameters.amendments is not None:
            for cmd in self.parameters.amendments:
                self.lmp.command(cmd)

        if n_steps > 0:
            if velocity_field is None:
                vel = convert(
                    atoms.get_velocities(),
                    "velocity",
                    "ASE",
                    self.units)
            else:
                # FIXME: Do we need to worry about converting to lammps units
                # here?
                vel = atoms.arrays[velocity_field]

            # If necessary, transform the velocities to new coordinate system
            if self.coord_transform is not None:
                vel = np.dot(self.coord_transform, vel.T).T

            # Convert ase velocities matrix to lammps-style velocities array
            lmp_velocities = list(vel.ravel())

            # Convert that lammps-style array into a C object
            c_double_array = (ctypes.c_double * len(lmp_velocities))
            lmp_c_velocities = c_double_array(*lmp_velocities)
            self.lmp.scatter_atoms('v', 1, 3, lmp_c_velocities)

        # Run for 0 time to calculate
        if dt is not None:
            if dt_not_real_time:
                self.lmp.command('timestep {:.30f}'.format(dt))
            else:
                self.lmp.command('timestep {:.30f}'.format(convert(dt, "time", "ASE", self.units)))
        self.lmp.command('run {:d}'.format(n_steps))

        if n_steps > 0:
            # TODO this must be slower than native copy, but why is it broken?
            pos = np.array(
                [x for x in self.lmp.gather_atoms("x", 1, 3)]).reshape(-1, 3)
            if self.coord_transform is not None:
                pos = np.dot(pos, self.coord_transform)

            # Convert from LAMMPS units to ASE units
            pos = convert(pos, "distance", self.units, "ASE")

            atoms.set_positions(pos)

            vel = np.array(
                [v for v in self.lmp.gather_atoms("v", 1, 3)]).reshape(-1, 3)
            if self.coord_transform is not None:
                vel = np.dot(vel, self.coord_transform)
            if velocity_field is None:
                atoms.set_velocities(convert(vel, 'velocity', self.units,
                                             'ASE'))

        # Extract the forces and energy
        self.results['energy'] = convert(
            self.lmp.extract_variable('pe', None, 0),
            "energy", self.units, "ASE"
        )
        self.results['free_energy'] = self.results['energy']

        stress = np.empty(6)
        stress_vars = ['pxx', 'pyy', 'pzz', 'pyz', 'pxz', 'pxy']

        for i, var in enumerate(stress_vars):
            stress[i] = self.lmp.extract_variable(var, None, 0)

        stress_mat = np.zeros((3, 3))
        stress_mat[0, 0] = stress[0]
        stress_mat[1, 1] = stress[1]
        stress_mat[2, 2] = stress[2]
        stress_mat[1, 2] = stress[3]
        stress_mat[2, 1] = stress[3]
        stress_mat[0, 2] = stress[4]
        stress_mat[2, 0] = stress[4]
        stress_mat[0, 1] = stress[5]
        stress_mat[1, 0] = stress[5]
        if self.coord_transform is not None:
            stress_mat = np.dot(self.coord_transform.T,
                                np.dot(stress_mat, self.coord_transform))
        stress[0] = stress_mat[0, 0]
        stress[1] = stress_mat[1, 1]
        stress[2] = stress_mat[2, 2]
        stress[3] = stress_mat[1, 2]
        stress[4] = stress_mat[0, 2]
        stress[5] = stress_mat[0, 1]

        self.results['stress'] = convert(-stress, "pressure", self.units, "ASE")

        # definitely yields atom-id ordered force array
        f = convert(np.array(self.lmp.gather_atoms("f", 1, 3)).reshape(-1, 3),
                    "force", self.units, "ASE")

        if self.coord_transform is not None:
            self.results['forces'] = np.dot(f, self.coord_transform)
        else:
            self.results['forces'] = f.copy()

        # otherwise check_state will always trigger a new calculation
        self.atoms = atoms.copy()

        if not self.parameters.keep_alive:
            self.lmp.close()

    def lammpsbc(self, atoms):
        """Determine LAMMPS boundary types based on ASE pbc settings. For
        non-periodic dimensions, if the cell length is finite then
        fixed BCs ('f') are used; if the cell length is approximately
        zero, shrink-wrapped BCs ('s') are used."""

        retval = ''
        pbc = atoms.get_pbc()
        if np.all(pbc):
            retval = 'p p p'
        else:
            cell = atoms.get_cell()
            for i in range(0, 3):
                if pbc[i]:
                    retval += 'p '
                else:
                    # See if we're using indefinite ASE boundaries along this
                    # direction
                    if np.linalg.norm(cell[i]) < np.finfo(cell[i][0]).tiny:
                        retval += 's '
                    else:
                        retval += 'f '

        return retval.strip()

    def rebuild(self, atoms):
        try:
            n_diff = len(atoms.numbers) - len(self.previous_atoms_numbers)
        except Exception:  # XXX Which kind of exception?
            n_diff = len(atoms.numbers)

        if n_diff > 0:
            if any([("reaxff" in cmd) for cmd in self.parameters.lmpcmds]):
                self.lmp.command("pair_style lj/cut 2.5")
                self.lmp.command("pair_coeff * * 1 1")

                for cmd in self.parameters.lmpcmds:
                    if (("pair_style" in cmd) or ("pair_coeff" in cmd) or
                            ("qeq/reaxff" in cmd)):
                        self.lmp.command(cmd)

            cmd = "create_atoms 1 random {} 1 NULL".format(n_diff)
            self.lmp.command(cmd)
        elif n_diff < 0:
            cmd = "group delatoms id {}:{}".format(
                len(atoms.numbers) + 1, len(self.previous_atoms_numbers))
            self.lmp.command(cmd)
            cmd = "delete_atoms group delatoms"
            self.lmp.command(cmd)

        self.redo_atom_types(atoms)

    def redo_atom_types(self, atoms):
        current_types = set(
            (i + 1, self.parameters.atom_types[sym]) for i, sym
            in enumerate(atoms.get_chemical_symbols()))

        try:
            previous_types = set(
                (i + 1, self.parameters.atom_types[ase_chemical_symbols[Z]])
                for i, Z in enumerate(self.previous_atoms_numbers))
        except Exception:  # XXX which kind of exception?
            previous_types = set()

        for (i, i_type) in current_types - previous_types:
            cmd = "set atom {} type {}".format(i, i_type)
            self.lmp.command(cmd)

        self.previous_atoms_numbers = atoms.numbers.copy()

    def restart_lammps(self, atoms):
        if self.started:
            self.lmp.command("clear")
        # hope there's no other state to be reset
        self.started = False
        self.initialized = False
        self.previous_atoms_numbers = []
        self.start_lammps()
        self.initialise_lammps(atoms)

    def start_lammps(self):
        # Only import lammps when running a calculation
        # so it is not required to use other parts of the
        # module
        from lammps import lammps
        # start lammps process
        if self.parameters.log_file is None:
            cmd_args = ['-echo', 'log', '-log', 'none', '-screen', 'none',
                        '-nocite']
        else:
            cmd_args = ['-echo', 'log', '-log', self.parameters.log_file,
                        '-screen', 'none', '-nocite']

        self.cmd_args = cmd_args

        if self.lmp is None:
            self.lmp = lammps(self.parameters.lammps_name, self.cmd_args,
                              comm=self.parameters.comm)

        # Run header commands to set up lammps (units, etc.)
        for cmd in self.parameters.lammps_header:
            self.lmp.command(cmd)

        for cmd in self.parameters.lammps_header:
            if "units" in cmd:
                self.units = cmd.split()[1]

        if 'lammps_header_extra' in self.parameters:
            if self.parameters.lammps_header_extra is not None:
                for cmd in self.parameters.lammps_header_extra:
                    self.lmp.command(cmd)

        self.started = True

    def initialise_lammps(self, atoms):
        # Initialising commands
        if self.parameters.boundary:
            # if the boundary command is in the supplied commands use that
            # otherwise use atoms pbc
            for cmd in self.parameters.lmpcmds:
                if 'boundary' in cmd:
                    break
            else:
                self.lmp.command('boundary ' + self.lammpsbc(atoms))

        # Initialize cell
        self.set_cell(atoms, change=not self.parameters.create_box)

        if self.parameters.atom_types is None:
            # if None is given, create from atoms object in order of appearance
            s = atoms.get_chemical_symbols()
            _, idx = np.unique(s, return_index=True)
            s_red = np.array(s)[np.sort(idx)].tolist()
            self.parameters.atom_types = {j: i + 1 for i, j in enumerate(s_red)}

        # Initialize box
        if self.parameters.create_box:
            # count number of known types
            n_types = len(self.parameters.atom_types)
            create_box_command = 'create_box {} cell'.format(n_types)
            self.lmp.command(create_box_command)

        # Initialize the atoms with their types
        # positions do not matter here
        if self.parameters.create_atoms:
            self.lmp.command('echo none')  # don't echo the atom positions
            self.rebuild(atoms)
            self.lmp.command('echo log')  # turn back on
        else:
            self.previous_atoms_numbers = atoms.numbers.copy()

        # execute the user commands
        for cmd in self.parameters.lmpcmds:
            self.lmp.command(cmd)

        # Set masses after user commands, e.g. to override
        # EAM-provided masses
        for sym in self.parameters.atom_types:
            if self.parameters.atom_type_masses is None:
                mass = ase_atomic_masses[ase_atomic_numbers[sym]]
            else:
                mass = self.parameters.atom_type_masses[sym]
            self.lmp.command('mass %d %.30f' % (
                self.parameters.atom_types[sym],
                convert(mass, "mass", "ASE", self.units)))

        # Define force & energy variables for extraction
        self.lmp.command('variable pxx equal pxx')
        self.lmp.command('variable pyy equal pyy')
        self.lmp.command('variable pzz equal pzz')
        self.lmp.command('variable pxy equal pxy')
        self.lmp.command('variable pxz equal pxz')
        self.lmp.command('variable pyz equal pyz')

        # I am not sure why we need this next line but LAMMPS will
        # raise an error if it is not there. Perhaps it is needed to
        # ensure the cell stresses are calculated
        self.lmp.command('thermo_style custom pe pxx emol ecoul')
        self.lmp.command('variable fx atom fx')
        self.lmp.command('variable fy atom fy')
        self.lmp.command('variable fz atom fz')
        self.lmp.command('variable pe equal pe')
        self.lmp.command("neigh_modify delay 0 every 1 check yes")
        self.initialized  = True

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

    def __del__(self):
        if self.started:
            self.lmp.close()
            self.started = False
            self.lmp = None
