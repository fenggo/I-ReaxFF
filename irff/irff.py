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
from .reaxfflib import read_ffield,write_lib
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

    def __init__(self,atoms=None,
                mol=None,
                libfile='ffield',
                vdwcut=10.0,
                atol=0.001,
                hbtol=0.001,
                hbshort=6.75,hblong=7.5,
                label="IRFF", 
                *args,**kwargs):
        Calculator.__init__(self,label=label, **kwargs)
        self.atoms        = atoms
        self.cell         = atoms.get_cell()
        self.atom_name    = self.atoms.get_chemical_symbols()
        self.natom        = len(self.atom_name)
        self.spec         = []
        self.nn           = nn
        # self.vdwnn      = vdwnn
        self.EnergyFunction = 0
        self.autograd     = autograd
        self.GPa          = 1.60217662*1.0e2
        # self.CalStress    = CalStress

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
            self.init_bonds()
            if mol is None:
                self.emol = 0.0
            else:
                mol_ = mol.split('-')[0]
                if mol_ in self.MolEnergy_:
                self.emol = self.MolEnergy_[mol_]
                else:
                self.emol = 0.0

            self.spec,self.bonds,self.offd,self.angs,self.torp,self.hbs = init_bonds(self.p)
            write_ffield(self.p,self.spec,self.bonds,self.offd,self.angs,self.torp,self.hbs,
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
            self.vdwnn    = False
            self.EnergyFunction  = 0
            self.MessageFunction = 0
            self.VdwFunction     = 0
            self.p['acut']   = 0.0001
            self.p['hbtol']  = 0.0001
        if m is None:
            self.nn=False
            
        for sp in self.atom_name:
            if sp not in self.spec:
                self.spec.append(sp)

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

    def calculate(self, atoms, properties, system_changes):
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
        self.initialized = True

    def __del__(self):
        if self.started:
            self.lmp.close()
            self.started = False
            self.lmp = None
