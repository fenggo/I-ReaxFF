from os import system,getcwd,chdir 
from os.path import isfile,exists
import operator as op
from collections import OrderedDict
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.espresso import read_espresso_in # ,read_espresso_out
from ase.constraints import FixAtoms, FixCartesian
from ase.units import create_units,Ry
from ase.data import chemical_symbols, atomic_numbers
from ase.dft.kpoints import kpoint_convert
from ase.calculators.singlepoint import SinglePointDFTCalculator,SinglePointKPoint
from ase.atoms import Atoms
import numpy as np
'''  tools used by QE '''


def single_point(atoms,id=0,kpts=(1,1,1),
                 val={'C':4.0,'H':1.0,'O':6.0,'N':5.0,'F':7.0,'Al':3.0},
                 cpu=4,**kwargs):

    write_qe_in(atoms,kpts=kpts,koffset=(0, 0, 0),**kwargs) # for MD
    system('mpirun -n %d pw.x<pw.in>pw.out' %cpu)

    images  = read_espresso_out('pw.out')
    system('cp pw.out pw-{:d}.out'.format(id))
    
    atoms_  = None
    for atoms_ in images:
        e = atoms_.get_potential_energy()
    return atoms_

def qemd(atoms=None,gen='poscar.gen',kpts=(1,1,1),
         label='pw',ncpu=1,T=300,dt=0.1,tstep=10):
    if atoms is None:
       atoms = read(gen)
    write_qe_in(atoms,ion_dynamics='bfgs',
                calculation='md',T=300,
                kpts=kpts,koffset=(0, 0, 0)) # for MD
    system('mpirun -n %d pw.x<pw.in>pw.out' %ncpu)

    his = TrajectoryWriter(label+'.traj',mode='w')
    atoms  = read_espresso_out('pw.out')
    images = []
    for atoms_ in atoms:
        images.append(atoms_)
        his.write(atoms=atoms_)
    his.close()
    return images

def qeopt(atoms=None,gen='poscar.gen',kpts=(1,1,1),
          label='pw',ncpu=1):
    if atoms is None:
       atoms = read(gen)
    write_qe_in(atoms,ion_dynamics='bfgs',
                calculation='relax',
                kpts=kpts,koffset=(0, 0, 0)) # for structure relax
    system('mpirun -n %d pw.x<pw.in>pw.out' %ncpu)

    his = TrajectoryWriter(label+'.traj',mode='w')
    atoms  = read_espresso_out('pw.out')
    images = []
    for atoms_ in atoms:
        images.append(atoms_)
        his.write(atoms=atoms_)
    his.close()
    return images

def run_qe(ncpu=12):
    system('mpirun -n %d pw.x<in.fdf>pw.out' %ncpu)

def pseudopotentials(specie):
    return specie+'.UPF'

def write_qe_in(atoms,
                kspacing=None, kpts=None,koffset=(0, 0, 0),
                calculation=None,          # supported: 'md','relax'
                forc_conv_thr=0.0001,
                ion_dynamics='bfgs',
                nspin=1,fspin=None,**kwargs):
    ''' wtrite QE(PWscf) input'''
    constraint_mask = np.ones((len(atoms), 3), dtype='int')
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            constraint_mask[constraint.index] = 0
        elif isinstance(constraint, FixCartesian):
            constraint_mask[constraint.a] = constraint.mask
        else:
            warnings.warn('Ignored unknown constraint {}'.format(constraint))

    atomic_species       = {} # atoms.get_chemical_symbols()
    atomic_species_str   = []
    atomic_positions_str = []
    mag_str = []

    if 'magmoms' in kwargs:
       atoms.set_initial_magnetic_moments(magmoms=kwargs['magmoms'])

    if any(atoms.get_initial_magnetic_moments()):
        if nspin == 1:
            # Force spin on
            nspin = 2

    if nspin == 2:
        # Spin on
        for atom, magmom in zip(atoms, atoms.get_initial_magnetic_moments()):
            #print(atomic_species)
            if (atom.symbol, magmom) not in atomic_species:
               # spin as fraction of valence
               # fspin = float(magmom) / species_info[atom.symbol]['valence']
               # Index in the atomic species list
               sidx = len(atomic_species) + 1
               # Index for that atom type; no index for first one
               tidx = sum(atom.symbol == x[0] for x in atomic_species) or ' '
               atomic_species[(atom.symbol, magmom)] = (sidx, tidx)
               # Add magnetization to the input file
               # print(sidx,magmom)
               mag_str.append('starting_magnetization({:d})  = {:7.4f}'.format(sidx,magmom))
               
               #input_parameters['system'][mag_str] = fspin
               atomic_species_str.append(
                  '{species}{tidx} {mass} {pseudo}\n'.format(
                        species=atom.symbol, tidx=tidx, mass=atom.mass,
                        pseudo=pseudopotentials(atom.symbol)))
            # lookup tidx to append to name
            sidx, tidx = atomic_species[(atom.symbol, magmom)]

            # only inclued mask if something is fixed
            if not all(constraint_mask[atom.index]):
               mask = ' {mask[0]} {mask[1]} {mask[2]}'.format(
                    mask=constraint_mask[atom.index])
            else:
               mask = ''

            # construct line for atomic positions
            atomic_positions_str.append(
                '{atom.symbol}{tidx} '
                '{atom.x:.10f} {atom.y:.10f} {atom.z:.10f}'
                '{mask}\n'.format(atom=atom, tidx=tidx, mask=mask))
        #for sp in 
        
    else:
        for atom in atoms:
            if atom.symbol not in atomic_species:
               atomic_species[atom.symbol] = True  # just a placeholder
               atomic_species_str.append(
                   '{species} {mass} {pseudo}\n'.format(
                       species=atom.symbol, mass=atom.mass,
                       pseudo=pseudopotentials(atom.symbol)))

           # only inclued mask if something is fixed
            if not all(constraint_mask[atom.index]):
               mask = ' {mask[0]} {mask[1]} {mask[2]}'.format(
                   mask=constraint_mask[atom.index])
            else:
               mask = ''

            atomic_positions_str.append(
               '{atom.symbol} '
               '{atom.x:.10f} {atom.y:.10f} {atom.z:.10f} '
               '{mask}\n'.format(atom=atom, mask=mask))
    # print(mag_str)
    # Add computed parameters
    # different magnetisms means different types
    # input_parameters['system']['ntyp'] = len(atomic_species)
    # input_parameters['system']['nat'] = len(atoms)
    # Construct input file into this
    pwi = []

    pwi.append('&CONTROL\n')
    if not calculation is None:
       pwi.append('  calculation     =  \'{:s}\' \n'.format(calculation))
    pwi.append('  outdir          =  \'./\' \n')
    pwi.append('  pseudo_dir      =  \'./\' \n')
    pwi.append('  wfcdir          =  \'./\' \n')
    pwi.append('  forc_conv_thr   =  {:f}\n'.format(forc_conv_thr))
    pwi.append('/\n')  # terminate section
    pwi.append('\n')

    pwi.append('&SYSTEM\n')
    if 'ibrav' in kwargs:
       pwi.append('  ibrav           =  {:f} \n'.format(kwargs['ibrav']))
    else:
       pwi.append('  ibrav           =  0 \n')
    pwi.append('  nat             =  {:d} \n'.format(len(atoms)))
    pwi.append('  ntyp            =  {:d} \n'.format(len(atomic_species)))
    if 'ecutwfc' in kwargs:
       pwi.append('  ecutwfc         =  {:f} \n'.format(kwargs['ecutwfc']))
    else:
       pwi.append('  ecutwfc         =  50 \n')
    if 'ecutrho' in kwargs:
       pwi.append('  ecutrho         =  {:f} \n'.format(kwargs['ecutrho']))
    else:
       pwi.append('  ecutrho         =  600 \n')
    pwi.append('  occupations     =  \'smearing\' \n')
    pwi.append('  smearing        =  \'gauss\' \n') # methfessel-paxton
    pwi.append('  degauss         =  0.2          !defult 0.D0 Ry \n')

    if nspin==2:
       for m in mag_str:
           m_ = float(m.split()[-1])
           if abs(m_)>0.00001:
              pwi.append('  '+m+'\n')
    if 'lda_plus_u' in kwargs:
       pwi.append('  lda_plus_u   =  .true. \n')
    if 'hubbard_u' in kwargs:
       hubbard_u = kwargs['hubbard_u']
       for u in hubbard_u:
           pwi.append('  hubbard_u({:d})   =  {:6.4f} \n'.format(u,hubbard_u[u]))
    pwi.append('/\n')  # terminate section
    pwi.append('\n')

    pwi.append('&ELECTRONS\n')
    if 'conv_thr' in kwargs:
       pwi.append('  conv_thr         =  {:f} \n'.format(kwargs['conv_thr']))
    else:
       pwi.append('  conv_thr         =  1.00000e-06           !defult 1.0d-6 \n')
    if 'mixing_beta' in kwargs:
       pwi.append('  mixing_beta      =  {:f} \n'.format(kwargs['mixing_beta']))
    else:
       pwi.append('  mixing_beta      =  7.00000e-01           !defult 0.7D0 \n')
    pwi.append('/\n')  # terminate section
    pwi.append('\n')

    if calculation == 'md':
       pwi.append('&IONS\n')
       pwi.append('  ion_temperature = \'not_controlled\' \n')
       pwi.append('  tempw           = \'{:.4f}\' \n'.format(T))
       pwi.append('/\n')  # terminate section
       pwi.append('\n')
    elif calculation == 'relax':
       pwi.append('&IONS\n')
       pwi.append('  ion_dynamics    = \'bfgs\' \n')
       pwi.append('/\n')  # terminate section
       pwi.append('\n')
    # pwi.append('&CELL\n')
    # pwi.append('/\n')  # terminate section
    # pwi.append('\n')
    # True and False work here and will get converted by ':d' format
    # KPOINTS - add a MP grid as required
    
    if kspacing is not None:
        kgrid = kspacing_to_grid(atoms, kspacing)
    elif kpts is not None:
        if isinstance(kpts, dict) and 'path' not in kpts:
            kgrid, shift = kpts2sizeandoffsets(atoms=atoms, **kpts)
            koffset = []
            for i, x in enumerate(shift):
                assert x == 0 or abs(x * kgrid[i] - 0.5) < 1e-14
                koffset.append(0 if x == 0 else 1)
        else:
            kgrid = kpts
    else:
        kgrid = (1, 1, 1)

    # True and False work here and will get converted by ':d' format
    if isinstance(koffset, int):
        koffset = (koffset, ) * 3

    # CELL block, if required
    pwi.append('CELL_PARAMETERS angstrom\n')
    pwi.append('{cell[0][0]:.14f} {cell[0][1]:.14f} {cell[0][2]:.14f}\n'
               '{cell[1][0]:.14f} {cell[1][1]:.14f} {cell[1][2]:.14f}\n'
               '{cell[2][0]:.14f} {cell[2][1]:.14f} {cell[2][2]:.14f}\n'
               ''.format(cell=atoms.cell))
    pwi.append('\n')

    # Pseudopotentials
    pwi.append('ATOMIC_SPECIES\n')
    pwi.extend(atomic_species_str)
    pwi.append('\n')

    # Positions - already constructed, but must appear after namelist
    pwi.append('ATOMIC_POSITIONS angstrom\n')
    pwi.extend(atomic_positions_str)
    pwi.append('\n')

    # BandPath object or bandpath-as-dictionary:
    if 'K_POINTS' in kwargs:
       pwi.append('K_POINTS automatic\n')
       pwi.append(kwargs['K_POINTS'])
    else:
       pwi.append('K_POINTS automatic\n')
       pwi.append('{0[0]} {0[1]} {0[2]}  {1[0]:d} {1[1]:d} {1[2]:d}\n'
                  ''.format(kgrid, koffset))
    pwi.append('\n')

    with open('pw.in','w') as fd:
         fd.write(''.join(pwi))

def read_espresso_out(fileo):
    with open(fileo, 'rU') as fileobj:
         pwo_lines = fileobj.readlines() # work with a copy in memory for faster random access

    # Quantum ESPRESSO uses CODATA 2006 internally
    units = create_units('2006')

    # Section identifiers
    _PW_START = 'Program PWSCF'
    _PW_END = 'End of self-consistent calculation'
    _PW_CELL = 'CELL_PARAMETERS'
    _PW_POS = 'ATOMIC_POSITIONS'
    _PW_MAGMOM = 'Magnetic moment per site'
    _PW_FORCE = 'Forces acting on atoms'
    _PW_TOTEN = '!    total energy'
    _PW_STRESS = 'total   stress'
    _PW_FERMI = 'the Fermi energy is'
    _PW_HIGHEST_OCCUPIED = 'highest occupied level'
    _PW_HIGHEST_OCCUPIED_LOWEST_FREE = 'highest occupied, lowest unoccupied level'
    _PW_KPTS = 'number of k points='
    _PW_BANDS = _PW_END
    _PW_BANDSTRUCTURE = 'End of band structure calculation'

    indexes = {
        _PW_START: [],
        _PW_END: [],
        _PW_CELL: [],
        _PW_POS: [],
        _PW_MAGMOM: [],
        _PW_FORCE: [],
        _PW_TOTEN: [],
        _PW_STRESS: [],
        _PW_FERMI: [],
        _PW_HIGHEST_OCCUPIED: [],
        _PW_HIGHEST_OCCUPIED_LOWEST_FREE: [],
        _PW_KPTS: [],
        _PW_BANDS: [],
        _PW_BANDSTRUCTURE: [],
    }

    for idx, line in enumerate(pwo_lines):
        for identifier in indexes:
            if identifier in line:
                indexes[identifier].append(idx)

    # Configurations are either at the start, or defined in ATOMIC_POSITIONS
    # in a subsequent step. Can deal with concatenated output files.
    all_config_indexes = sorted(indexes[_PW_START] +
                                indexes[_PW_POS])

    results_indexes = sorted(indexes[_PW_TOTEN] + indexes[_PW_FORCE] +
                             indexes[_PW_STRESS] + indexes[_PW_MAGMOM] +
                             indexes[_PW_BANDS] +
                             indexes[_PW_BANDSTRUCTURE])
    # Prune to only configurations with results data before the next
    # configuration
    image_indexes = []
    for config_index, config_index_next in zip(
             all_config_indexes,
             all_config_indexes[1:] + [len(pwo_lines)]):
        # print(config_index, config_index_next)
        if any([config_index < results_index < config_index_next
                 for results_index in results_indexes]):
           image_indexes.append(config_index)

    # Extract initialisation information each time PWSCF starts
    # to add to subsequent configurations. Use None so slices know
    # when to fill in the blanks.
    pwscf_start_info = dict((idx, None) for idx in indexes[_PW_START])

    for image_index in image_indexes:
        # Find the nearest calculation start to parse info. Needed in,
        # for example, relaxation where cell is only printed at the
        # start.
        if image_index in indexes[_PW_START]:
            prev_start_index = image_index
        else:
            # The greatest start index before this structure
            prev_start_index = [idx for idx in indexes[_PW_START]
                                if idx < image_index][-1]

        # add structure to reference if not there
        if pwscf_start_info[prev_start_index] is None:
            pwscf_start_info[prev_start_index] = parse_pwo_start(
                pwo_lines, prev_start_index)

        # Get the bounds for information for this structure. Any associated
        # values will be between the image_index and the following one,
        # EXCEPT for cell, which will be 4 lines before if it exists.
        for next_index in all_config_indexes:
            if next_index > image_index:
                break
        else:
            # right to the end of the file
            next_index = len(pwo_lines)

        # Get the structure
        # Use this for any missing data
        prev_structure = pwscf_start_info[prev_start_index]['atoms']
        if image_index in indexes[_PW_START]:
            structure = prev_structure.copy()  # parsed from start info
        else:
            if _PW_CELL in pwo_lines[image_index - 5]:
                # CELL_PARAMETERS would be just before positions if present
                cell, cell_alat = get_cell_parameters(
                    pwo_lines[image_index - 5:image_index])
            else:
                cell = prev_structure.cell
                cell_alat = pwscf_start_info[prev_start_index]['alat']

            # give at least enough lines to parse the positions
            # should be same format as input card
            n_atoms = len(prev_structure)
            positions_card = get_atomic_positions(
                pwo_lines[image_index:image_index + n_atoms + 1],
                n_atoms=n_atoms, cell=cell, alat=cell_alat)

            # convert to Atoms object
            symbols = [label_to_symbol(position[0]) for position in
                       positions_card]
            positions = [position[1] for position in positions_card]

            constraint_idx = [position[2] for position in positions_card]
            constraint = get_constraint(constraint_idx)

            structure = Atoms(symbols=symbols, positions=positions, cell=cell,
                              constraint=constraint, pbc=True)

        # Extract calculation results
        # Energy
        energy = None
        for energy_index in indexes[_PW_TOTEN]:
            if image_index < energy_index < next_index:
                energy = float(
                    pwo_lines[energy_index].split()[-2]) * units['Ry']

        # Forces
        forces = None
        for force_index in indexes[_PW_FORCE]:
            if image_index < force_index < next_index:
                # Before QE 5.3 'negative rho' added 2 lines before forces
                # Use exact lines to stop before 'non-local' forces
                # in high verbosity
                if not pwo_lines[force_index + 2].strip():
                    force_index += 4
                else:
                    force_index += 2
                # assume contiguous
                forces = [
                    [float(x) for x in force_line.split()[-3:]] for force_line
                    in pwo_lines[force_index:force_index + len(structure)]]
                forces = np.array(forces) * units['Ry'] / units['Bohr']

        # Stress
        stress = None
        for stress_index in indexes[_PW_STRESS]:
            if image_index < stress_index < next_index:
                sxx, sxy, sxz = pwo_lines[stress_index + 1].split()[:3]
                _, syy, syz = pwo_lines[stress_index + 2].split()[:3]
                _, _, szz = pwo_lines[stress_index + 3].split()[:3]
                stress = np.array([sxx, syy, szz, syz, sxz, sxy], dtype=float)
                # sign convention is opposite of ase
                stress *= -1 * units['Ry'] / (units['Bohr'] ** 3)

        # Magmoms
        magmoms = None
        for magmoms_index in indexes[_PW_MAGMOM]:
            if image_index < magmoms_index < next_index:
                magmoms = [
                    float(mag_line.split()[6]) for mag_line
                    in pwo_lines[magmoms_index + 1:
                                 magmoms_index + 1 + len(structure)]]

        # Fermi level / highest occupied level
        efermi = None
        for fermi_index in indexes[_PW_FERMI]:
            if image_index < fermi_index < next_index:
                efermi = float(pwo_lines[fermi_index].split()[-2])

        if efermi is None:
            for ho_index in indexes[_PW_HIGHEST_OCCUPIED]:
                if image_index < ho_index < next_index:
                    efermi = float(pwo_lines[ho_index].split()[-1])

        if efermi is None:
            for holf_index in indexes[_PW_HIGHEST_OCCUPIED_LOWEST_FREE]:
                if image_index < holf_index < next_index:
                    efermi = float(pwo_lines[holf_index].split()[-2])

        # K-points
        ibzkpts = None
        weights = None
        kpoints_warning = "Number of k-points >= 100: " + \
                          "set verbosity='high' to print them."

        for kpts_index in indexes[_PW_KPTS]:
            nkpts = int(pwo_lines[kpts_index].split()[4])
            kpts_index += 2

            if pwo_lines[kpts_index].strip() == kpoints_warning:
                continue

            # QE prints the k-points in units of 2*pi/alat
            # with alat defined as the length of the first
            # cell vector
            cell = structure.get_cell()
            alat = np.linalg.norm(cell[0])
            ibzkpts = []
            weights = []
            for i in range(nkpts):
                l = pwo_lines[kpts_index + i].split()
                weights.append(float(l[-1]))
                coord = np.array([l[-6], l[-5], l[-4].strip('),')],
                                 dtype=float)
                coord *= 2 * np.pi / alat
                coord = kpoint_convert(cell, ckpts_kv=coord)
                ibzkpts.append(coord)
            ibzkpts = np.array(ibzkpts)
            weights = np.array(weights)

        # Bands
        kpts = None
        kpoints_warning = "Number of k-points >= 100: " + \
                          "set verbosity='high' to print the bands."

        # for bands_index in indexes[_PW_BANDS] + indexes[_PW_BANDSTRUCTURE]:
        #     if image_index < bands_index < next_index:
        #         bands_index += 2

        #         if pwo_lines[bands_index].strip() == kpoints_warning:
        #             continue

        #         assert ibzkpts is not None
        #         spin, bands, eigenvalues = 0, [], [[], []]

        #         while True:
        #             l = pwo_lines[bands_index].replace('-', ' -').split()
        #             if len(l) == 0:
        #                 if len(bands) > 0:
        #                     eigenvalues[spin].append(bands)
        #                     bands = []
        #             elif l == ['occupation', 'numbers']:
        #                 # Skip the lines with the occupation numbers
        #                 bands_index += len(eigenvalues[spin][0]) // 8 + 1
        #             elif l[0] == 'k' and l[1].startswith('='):
        #                 pass
        #             elif 'SPIN' in l:
        #                 if 'DOWN' in l:
        #                     spin += 1
        #             else:
        #                 try:
        #                     bands.extend(map(float, l))
        #                 except ValueError:
        #                     break
        #             bands_index += 1

                # if spin == 1:
                #     assert len(eigenvalues[0]) == len(eigenvalues[1])
                # assert len(eigenvalues[0]) == len(ibzkpts), (np.shape(eigenvalues), len(ibzkpts))

                # kpts = []
                # for s in range(spin + 1):
                #     for w, k, e in zip(weights, ibzkpts, eigenvalues[s]):
                #         kpt = SinglePointKPoint(w, s, k, eps_n=e)
                #         kpts.append(kpt)

        # Put everything together
        calc = SinglePointDFTCalculator(structure, energy=energy,
                                        forces=forces, stress=stress)
                                        # magmoms=magmoms, efermi=efermi,
                                        # ibzkpts=ibzkpts)
        calc.kpts = kpts
        structure.set_calculator(calc)

        yield structure

def get_constraint(constraint_idx):
    """
    Map constraints from QE input/output to FixAtoms or FixCartesian constraint
    """
    if not np.any(constraint_idx):
        return None

    a = [a for a, c in enumerate(constraint_idx) if np.all(c is not None)]
    mask = [[(ic + 1) % 2 for ic in c] for c in constraint_idx
            if np.all(c is not None)]

    if np.all(np.array(mask)) == 1:
        constraint = FixAtoms(a)
    else:
        constraint = FixCartesian(a, mask)
    return constraint

def parse_pwo_start(lines, index=0):
    units = create_units('2006')

    info = {}

    for idx, line in enumerate(lines[index:], start=index):
        if 'celldm(1)' in line:
            # celldm(1) has more digits than alat!!
            info['celldm(1)'] = float(line.split()[1]) * units['Bohr']
            info['alat'] = info['celldm(1)']
        elif 'number of atoms/cell' in line:
            info['nat'] = int(line.split()[-1])
        elif 'number of atomic types' in line:
            info['ntyp'] = int(line.split()[-1])
        elif 'crystal axes:' in line:
            info['cell'] = info['celldm(1)'] * np.array([
                [float(x) for x in lines[idx + 1].split()[3:6]],
                [float(x) for x in lines[idx + 2].split()[3:6]],
                [float(x) for x in lines[idx + 3].split()[3:6]]])
        elif 'positions (alat units)' in line:
            info['symbols'] = [
                label_to_symbol(at_line.split()[1])
                for at_line in lines[idx + 1:idx + 1 + info['nat']]]
            info['positions'] = [
                [float(x) * info['celldm(1)'] for x in at_line.split()[6:9]]
                for at_line in lines[idx + 1:idx + 1 + info['nat']]]
            # This should be the end of interesting info.
            # Break here to avoid dealing with large lists of kpoints.
            # Will need to be extended for DFTCalculator info.
            break

    # Make atoms for convenience
    info['atoms'] = Atoms(symbols=info['symbols'],
                          positions=info['positions'],
                          cell=info['cell'], pbc=True)
    return info

def label_to_symbol(label):
    if len(label) >= 2:
        test_symbol = label[0].upper() + label[1].lower()
        if test_symbol in chemical_symbols:
            return test_symbol
    # finally try with one character
    test_symbol = label[0].upper()
    if test_symbol in chemical_symbols:
        return test_symbol
    else:
        raise KeyError('Could not parse species from label {0}.'
                       ''.format(label))

def get_atomic_positions(lines, n_atoms, cell=None, alat=None):
    positions = None
    # no blanks or comment lines, can the consume n_atoms lines for positions
    trimmed_lines = (line for line in lines
                     if line.strip() and not line[0] == '#')

    for line in trimmed_lines:
        if line.strip().startswith('ATOMIC_POSITIONS'):
            if positions is not None:
                raise ValueError('Multiple ATOMIC_POSITIONS specified')
            # Priority and behaviour tested with QE 5.3
            if 'crystal_sg' in line.lower():
                raise NotImplementedError('CRYSTAL_SG not implemented')
            elif 'crystal' in line.lower():
                cell = cell
            elif 'bohr' in line.lower():
                cell = np.identity(3) * units['Bohr']
            elif 'angstrom' in line.lower():
                cell = np.identity(3)
            # elif 'alat' in line.lower():
            #     cell = np.identity(3) * alat
            else:
                if alat is None:
                    raise ValueError('Set lattice parameter in &SYSTEM for '
                                     'alat coordinates')
                # Always the default, will be DEPRECATED as mandatory
                # in future
                cell = np.identity(3) * alat

            positions = []
            for _dummy in range(n_atoms):
                split_line = next(trimmed_lines).split()
                # These can be fractions and other expressions
                position = np.dot((infix_float(split_line[1]),
                                   infix_float(split_line[2]),
                                   infix_float(split_line[3])), cell)
                if len(split_line) > 4:
                    force_mult = (float(split_line[4]),
                                  float(split_line[5]),
                                  float(split_line[6]))
                else:
                    force_mult = None

                positions.append((split_line[0], position, force_mult))
    return positions

def get_cell_parameters(lines, alat=None):
    cell = None
    cell_alat = None
    trimmed_lines = (line for line in lines
                     if line.strip() and not line[0] == '#')

    for line in trimmed_lines:
        if line.strip().startswith('CELL_PARAMETERS'):
            if cell is not None:
                # multiple definitions
                raise ValueError('CELL_PARAMETERS specified multiple times')
            # Priority and behaviour tested with QE 5.3
            if 'bohr' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in '
                                     '&SYSTEM celldm/A and CELL_PARAMETERS '
                                     'bohr')
                cell_units = units['Bohr']
            elif 'angstrom' in line.lower():
                if alat is not None:
                    raise ValueError('Lattice parameters given in '
                                     '&SYSTEM celldm/A and CELL_PARAMETERS '
                                     'angstrom')
                cell_units = 1.0
            elif 'alat' in line.lower():
                # Output file has (alat = value) (in Bohrs)
                if '=' in line:
                    alat = float(line.strip(') \n').split()[-1]) * units['Bohr']
                    cell_alat = alat
                elif alat is None:
                    raise ValueError('Lattice parameters must be set in '
                                     '&SYSTEM for alat units')
                cell_units = alat
            elif alat is None:
                # may be DEPRECATED in future
                cell_units = units['Bohr']
            else:
                # may be DEPRECATED in future
                cell_units = alat
            # Grab the parameters; blank lines have been removed
            cell = [[float(x) for x in next(trimmed_lines).split()[:3]],
                    [float(x) for x in next(trimmed_lines).split()[:3]],
                    [float(x) for x in next(trimmed_lines).split()[:3]]]
            cell = np.array(cell) * cell_units
    return cell, cell_alat

def infix_float(text):
    def middle_brackets(full_text):
        """Extract text from innermost brackets."""
        start, end = 0, len(full_text)
        for (idx, char) in enumerate(full_text):
            if char == '(':
                start = idx
            if char == ')':
                end = idx + 1
                break
        return full_text[start:end]

    def eval_no_bracket_expr(full_text):
        """Calculate value of a mathematical expression, no brackets."""
        exprs = [('+', op.add), ('*', op.mul),
                 ('/', op.truediv), ('^', op.pow)]
        full_text = full_text.lstrip('(').rstrip(')')
        try:
            return float(full_text)
        except ValueError:
            for symbol, func in exprs:
                if symbol in full_text:
                    left, right = full_text.split(symbol, 1)  # single split
                    return func(eval_no_bracket_expr(left),
                                eval_no_bracket_expr(right))

    while '(' in text:
        middle = middle_brackets(text)
        text = text.replace(middle, '{}'.format(eval_no_bracket_expr(middle)))

    return float(eval_no_bracket_expr(text))

