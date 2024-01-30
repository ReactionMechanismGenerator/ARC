# https://github.com/nmardirossian/PySCF_Tutorial/blob/master/user_guide.ipynb
import logging
from pyscf import gto, scf, dft
import pandas
import yaml
import os
import argparse
pandas.set_option('display.max_columns', 500)

def parse_command_line_arguments():
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(description='Run PySCF script with a YAML configuration file.')
    parser.add_argument('yml_path', type=str, help='Path to the YAML input file.')
    args = parser.parse_args()
    return args

class PYSCFScript_VB:
    
    def __init__(self, input_file) -> None:
        self.input_file = input_file
        self.input_dict = self.read_yaml_file()

        # Method
        self.method = self.get_method()
        # Molecule
        self.molecule = self.get_molecule()
        # Basis
        self.basis = self.get_basis()
        # Charge
        self.charge = self.get_charge()
        # Spin
        self.spin = self.get_spin()
        # Unit
        self.unit = self.get_unit()
        # Restricted
        self.restricted = self.get_restricted()
        # Conv tol
        self.conv_tol = self.get_conv_tol()
        # Conv Tol Grad
        self.conv_tol_grad = self.get_conv_tol_grad()
        # Direct scf tol
        self.direct_scf_tol = self.get_direct_scf_tol()
        # Init guess
        self.init_guess = self.get_init_guess()
        # Level Shift
        self.level_shift = self.get_level_shift()
        # Max cycles
        self.max_cycle = self.get_max_cycle()
        # Max memory
        self.max_memory = self.get_max_memory()
        # XC
        self.xc = self.get_xc()
        # NLC
        self.nlc = self.get_nlc()
        # xc grid
        self.xc_grid = self.get_xc_grid()
        # nlc grid
        self.nlc_grid = self.get_nlc_grid()
        # small_rho_cutoff
        self.small_rho_cutoff = self.get_small_rho_cutoff()
        # atomic radii
        self.atomic_radii = self.get_atomic_radii()
        # Becke scheme
        self.becke_scheme = self.get_becke_scheme()
        # Prune
        self.prune = self.get_prune()
        # Radi method
        self.radi_method = self.get_radi_method()
        # Radii Adjust
        self.radii_adjust = self.get_radii_adjust()
        # Algorithm
        self.algorithm = self.get_algorithm()
        # Datapath
        self.datapath = self.get_datapath()
        # getdict
        self.getdict = self.get_getdict()
        # lin depth
        self.lin_depth = self.get_lin_depth()
        # prop
        self.prop = self.get_prop()
        # scf type
        self.scf_type = self.get_scf_type()
        # Stable
        self.stable = self.get_stable()
        # Stable cyc
        self.stable_cyc = self.get_stable_cyc()
        # verbose
        self.verbose = self.get_verbose()
        # job type
        self.job_type = self.get_job_type()
        # is ts
        self.is_ts = self.get_ts_status()
        # maxsteps
        self.maxsteps = self.get_maxsteps()

        self.mol_eq = None
    def read_yaml_file(self):
        # Check if input file is a YAML file
        input_dict = None  # Initialize input_dict to ensure it's always defined

        try:
            if isinstance(self.input_file, str) and self.input_file.endswith('.yml'):
                with open(self.input_file, 'r') as f:
                    input_dict = yaml.load(f, Loader=yaml.FullLoader)
            elif isinstance(self.input_file, dict):
                input_dict = self.input_file
        except FileNotFoundError:
            raise FileNotFoundError('Could not find input file.') from FileNotFoundError

        # Check if input_dict was successfully populated
        if input_dict is None:
            raise ValueError("Input file is not valid or not a .yml file")

        return input_dict
    
    def get_method(self):
        # Get method in input dict if exists
        method = self.input_dict.get('xc_func', None)
        return method.upper()
    
    def get_molecule(self):
        # Get molecule in input dict if exists
        molecule = self.input_dict['xyz']
        return molecule
    
    def get_basis(self):
        # Get basis in input dict if exists
        basis = self.input_dict.get('basis', None)
        return basis
    
    def get_charge(self):
        # Get charge in input dict if exists
        charge = self.input_dict.get('charge', 0)
        return charge
    
    def get_spin(self):
        # Get spin in input dict if exists
        spin = self.input_dict.get('spin', 0)
        return spin
    
    def get_unit(self):
        # Get unit in input dict if exists
        # TODO: Should check the validity of the unit
        unit = self.input_dict.get('unit', 'ANG')
        return unit
    
    def get_restricted(self):
        # Get restricted in input dict if exists
        restricted = self.input_dict.get('restricted', True)
        return restricted
    
    def get_conv_tol(self):
        # Get conv_tol in input dict if exists
        conv_tol = self.input_dict.get('conv_tol', 1e-12)
        return conv_tol
    
    def get_conv_tol_grad(self):
        # Get conv_tol_grad in input dict if exists
        conv_tol_grad = self.input_dict.get('conv_tol_grad', 1e-8)
        return conv_tol_grad
    
    def get_direct_scf_tol(self):
        # Get direct_scf_tol in input dict if exists
        direct_scf_tol = self.input_dict.get('direct_scf_tol', 1e-13)
        return direct_scf_tol
    
    def get_init_guess(self):
        # Get init_guess in input dict if exists
        init_guess = self.input_dict.get('init_guess', 'minao')
        return init_guess
    
    def get_level_shift(self):
        # Get level_shift in input dict if exists
        level_shift = self.input_dict.get('level_shift', 0.0)
        return level_shift
    
    def get_max_cycle(self):
        # Get max_cycle in input dict if exists
        max_cycle = self.input_dict.get('max_cycle', 100)
        return max_cycle
    
    def get_max_memory(self):
        # Get max_memory in input dict if exists
        max_memory = self.input_dict.get('max_memory', 8000)
        return max_memory
    
    def get_xc(self):
        # Get xc in input dict if exists
        xc = self.input_dict.get('xc', None)
        return xc
    
    def get_nlc(self):
        # Get nlc in input dict if exists
        nlc = self.input_dict.get('nlc', None)
        return nlc
    
    def get_xc_grid(self):
        # Get xc_grid in input dict if exists
        xc_grid = self.input_dict.get('xc_grid', 3)
        return xc_grid
    
    def get_nlc_grid(self):
        # Get nlc_grid in input dict if exists
        nlc_grid = self.input_dict.get('nlc_grid', 3)
        return nlc_grid
    
    def get_small_rho_cutoff(self):
        # Get small_rho_cutoff in input dict if exists
        small_rho_cutoff = self.input_dict.get('small_rho_cutoff', 1e-7)
        return small_rho_cutoff
    
    def get_atomic_radii(self):
        # Get atomic_radii in input dict if exists
        atomic_radii = self.input_dict.get('atomic_radii', 'BRAGG')
        atomic_radii.upper()
        return atomic_radii
    
    def get_becke_scheme(self):
        # Get becke_scheme in input dict if exists
        becke_scheme = self.input_dict.get('becke_scheme', 'BECKE')
        becke_scheme.upper()
        return becke_scheme
    
    def get_prune(self):
        # Get prune in input dict if exists
        # scheme to reduce number of grids, can be one of
        # | gen_grid.nwchem_prune (default) | gen_grid.sg1_prune | gen_grid.treutler_prune | None : to switch off grid pruning
        prune = self.input_dict.get('prune', 'NWCHEM')
        if isinstance(prune, str):
            prune = prune.upper()
        return prune
    
    def get_radi_method(self):
        # Get radi_method in input dict if exists
        radi_method = self.input_dict.get('radi_method', 'TREUTLER_AHLRICHS')
        radi_method.upper()
        return radi_method
    
    def get_radii_adjust(self):
        # Get radii_adjust in input dict if exists
        radii_adjust = self.input_dict.get('radii_adjust', 'TREUTLER')
        if isinstance(radii_adjust, str):
            radii_adjust.upper()
        return radii_adjust
    
    def get_algorithm(self):
        # Get algorithm in input dict if exists
        algorithm = self.input_dict.get('algorithm', 'DIIS')
        algorithm.upper()
        return algorithm
    
    def get_datapath(self):
        # Get datapath in input dict if exists
        datapath = self.input_dict.get('datapath', None)
        return datapath
    
    def get_getdict(self):
        # Get getdict in input dict if exists
        getdict = self.input_dict.get('getdict', False)
        return getdict
    
    def get_lin_depth(self):
        # Get lin_depth in input dict if exists
        lin_depth = self.input_dict.get('lin_depth', 1e-8)
        return lin_depth
    
    def get_prop(self):
        # Get prop in input dict if exists
        prop = self.input_dict.get('prop', False)
        return prop
    
    def get_scf_type(self):
        # Get scf_type in input dict if exists
        scf_type = self.input_dict.get('scf_type', None)
        if scf_type:
            scf_type.upper()
        return scf_type
    
    def get_stable(self):
        # Get stable in input dict if exists
        stable = self.input_dict.get('stable', False)
        return stable
    
    def get_stable_cyc(self):
        # Get stable_cyc in input dict if exists
        stable_cyc = self.input_dict.get('stable_cyc', 4)
        return stable_cyc
    
    def get_verbose(self):
        # Get verbose in input dict if exists
        verbose = self.input_dict.get('verbose', 4)
        return verbose
    
    def get_job_type(self):
        job_type = self.input_dict['job_type']
        return job_type
    
    def get_ts_status(self):
        is_ts = self.input_dict['is_ts']
        if isinstance(is_ts, str):
            if is_ts.upper() == 'TRUE':
                is_ts = True
            elif is_ts.upper() == 'FALSE':
                is_ts = False
            else:
                raise ValueError('Invalid is_ts.')
        return is_ts
    
    def get_maxsteps(self):
        maxsteps = self.input_dict.get('maxsteps', 100)
        return maxsteps

    def read_molecule(self, path):

        charge = spin = 0
        with open(path, 'r') as myfile:
            output = myfile.read()
            output = output.lstrip()
            output = output.rstrip()
            output = output.split('\n')

        try:
            int(output[0])
        except ValueError:
            try:
                charge = int(output[0].split(' ')[0])
                spin = int(output[0].split(' ')[1]) - 1
            except ValueError:
                molecule = output
            else:
                molecule = '\n'.join(output[1:])
        else:
            if int(output[0]) == len(output) - 2:
                molecule = '\n'.join(output[2:])
                try:
                    charge = int(output[1].split(' ')[0])
                    spin = int(output[1].split(' ')[1])-1
                except ValueError:
                    pass
            else:
                print ("THIS IS NOT A VALID XYZ FILE")

        return (molecule, charge, spin)

    def convert_to_custom_xyz(self, atoms_and_coordinates):
        xyz_data = []
        xyz_data.append('\n')
        for index, coordinates in enumerate(atoms_and_coordinates):
            atom_line = f"{self.mol_eq.atom_symbol(index):<2}      {coordinates[0]: .8f}   {coordinates[1]: .8f}   {coordinates[2]: .8f}"
            xyz_data.append(atom_line)
        
        return '\n'.join(xyz_data)
    
    def output_to_yaml(self, w, modes):
        """
        Comber numpy arrays into a YAML file

        Args:
            w (_type_): _description_
            modes (_type_): _description_
        """
        
        freq = w.tolist()
        mode_list = [modes.tolist() for mode in modes]
        
        output_dict = {'freq': freq, 'modes': mode_list}
        
        with open(os.path.join(os.path.dirname(os.path.abspath(self.input_file)),'output_freq.yml'), 'w') as f:
            yaml.dump(output_dict, f)
    
    def run(self):
        
        # Set mol attributes
        mol = gto.Mole(atom=self.molecule, basis=self.basis, charge=self.charge, spin=self.spin, unit=self.unit).build()
        
        
        # Check method for density functional
        DFT = False
        if self.method != 'HF':
            try:
                # https://pyscf.org/_modules/pyscf/dft/libxc.html
                dft.libxc.parse_xc(self.method)
            except (ValueError, KeyError):
                raise ValueError(f"Method '{self.method}' is not recognized as HF or a valid DFT method.")
            else:
                # Kohn-Sham (KS)
                DFT = True
                self.xc = self.method
                self.method = 'KS'
                
        if self.method in ['HF', 'KS', 'DFT'] and self.scf_type is None:
            if self.spin == 0:
                self.scf_type = 'R'
            else:
                self.scf_type = 'U'
            
        # Create HF/KS/DFT object
        if self.method in ['RHF', 'ROHF'] or (self.method == 'HF' and self.scf_type in ['R', 'RO']):
            mf = scf.RHF(mol)
            self.scf_type = 'R'
        elif self.method == 'UHF' or (self.method == 'HF' and self.scf_type == 'U'):
            mf = scf.UHF(mol)
            self.scf_type = 'U'
        elif self.method in ['RKS', 'ROKS', 'RDFT', 'RODFT'] or (self.method in ['KS', 'DFT'] and self.scf_type in ['R', 'RO']):
            mf = dft.RKS(mol).density_fit()
            self.scf_type = 'R'
            DFT = True
        elif self.method in ['UKS', 'UDFT'] or (self.method in ['KS', 'DFT'] and self.scf_type == 'U'):
            mf = dft.UKS(mol).density_fit()
            self.scf_type = 'U'
            DFT = True
        else:
            raise ValueError('Invalid method.')
        
        # Set HF attributes

        mf.conv_check = True
        mf.conv_tol = self.conv_tol
        mf.conv_tol_grad = self.conv_tol_grad
        mf.direct_scf_tol = self.direct_scf_tol
        mf.init_guess = self.init_guess
        mf.level_shift = self.level_shift
        mf.max_cycle = self.max_cycle
        mf.max_memory = self.max_memory
        # mf.verbose = self.verbose
        
        # Set KS attributes
        if DFT:
            mf.xc = self.xc
            if mf.xc is None:
                raise ValueError('No XC functional specified.')
            #mf.nlc = self.nlc
            
            if isinstance(self.xc_grid, int):
                mf.grids.level = self.xc_grid
            elif isinstance(self.xc_grid, tuple) or isinstance(self.xc_grid, dict):
                mf.grids.atom_grid = self.xc_grid
            else:
                raise ValueError('Invalid xc_grid.')
            
            if isinstance(self.nlc_grid, int):
                mf.nlcgrids.level = self.nlc_grid
            elif isinstance(self.nlc_grid, tuple) or isinstance(self.nlc_grid, dict):
                mf.nlcgrids.atom_grid = self.nlc_grid
            else:
                raise ValueError('Invalid nlc_grid.')
            
            if self.atomic_radii == 'BRAGG':
                mf.grids.radi_method = dft.radi.BRAGG_RADII
                mf.nlcgrids.radi_method = dft.radi.BRAGG_RADII
            elif self.atomic_radii == 'COVALENT':
                mf.grids.radi_method = dft.radi.COVALENT_RADII
                mf.nlcgrids.radi_method = dft.radi.COVALENT_RADII
            else:
                raise ValueError('Invalid atomic_radii.')
            
            if self.becke_scheme == 'BECKE':
                mf.grids.becke_scheme = dft.gen_grid.original_becke
                mf.nlcgrids.becke_scheme = dft.gen_grid.original_becke
            elif self.becke_scheme == 'STRATMANN':
                mf.grids.becke_scheme = dft.gen_grid.stratmann
                mf.nlcgrids.becke_scheme = dft.gen_grid.stratmann
            else:
                raise ValueError('Invalid becke_scheme.')
            
            if self.prune == 'NWCHEM':
                mf.grids.prune = dft.gen_grid.nwchem_prune
                mf.nlcgrids.prune = dft.gen_grid.nwchem_prune
            elif self.prune == 'SG1':
                mf.grids.prune = dft.gen_grid.sg1_prune
                mf.nlcgrids.prune = dft.gen_grid.sg1_prune
            elif self.prune == 'TREUTLER':
                mf.grids.prune = dft.gen_grid.treutler_prune
                mf.nlcgrids.prune = dft.gen_grid.treutler_prune
            elif self.prune == 'NONE' or self.prune is None:
                mf.grids.prune = None
                mf.nlcgrids.prune = None
            else:
                raise ValueError('Invalid prune.')
            
            if self.radi_method in ['TREUTLER_AHLRICHS', 'TREUTLER', 'AHLRICHS']:
                mf.grids.radi_method = dft.radi.treutler_ahlrichs
                mf.nlcgrids.radi_method = dft.radi.treutler_ahlrichs
            elif self.radi_method == 'DELLEY':
                mf.grids.radi_method = dft.radi.delley
                mf.nlcgrids.radi_method = dft.radi.delley
            elif self.radi_method in ['MURA_KNOWLES', 'MURA', 'KNOWLES']:
                mf.grids.radi_method = dft.radi.mura_knowles
                mf.nlcgrids.radi_method = dft.radi.mura_knowles
            elif self.radi_method in ['GAUSS_CHEBYSHEV', 'GAUSS', 'CHEBYSHEV']:
                mf.grids.radi_method = dft.radi.gauss_chebyshev
                mf.nlcgrids.radi_method = dft.radi.gauss_chebyshev
            else:
                raise ValueError('Invalid radi_method.')
            
            if self.radii_adjust == 'TREUTLER':
                mf.grids.radii_adjust = dft.radi.treutler_atomic_radii_adjust
                mf.nlcgrids.radii_adjust = dft.radi.treutler_atomic_radii_adjust
            elif self.radii_adjust == 'BECKE':
                mf.grids.radii_adjust = dft.radi.becke_atomic_radii_adjust
                mf.nlcgrids.radii_adjust = dft.radi.becke_atomic_radii_adjust
            elif self.radii_adjust == 'NONE' or self.radii_adjust is None:
                mf.grids.radii_adjust = None
                mf.nlcgrids.radii_adjust = None
            else:
                raise ValueError('Invalid radii_adjust.')
            
            mf.small_rho_cutoff = self.small_rho_cutoff
        
        # Select Optimizer
        if self.algorithm == 'DIIS':
            mf.diis = True
        elif self.algorithm == 'EDIIS':
            mf.diis = scf.diis.EDIIS()
        elif self.algorithm == 'ADIIS':
            mf.diis = scf.diis.ADIIS()
        elif self.algorithm == 'NEWTON':
            mf = mf.newton()
        else:
            raise ValueError('Invalid algorithm.')       

        # # Running the Kernel
        
                # TODO: Check if TS
        if self.is_ts and (self.job_type == 'opt' or self.job_type == 'conformers'):
            from pyscf.qsdopt.qsd_optimizer import QSD
            #optimizer = QSD(mf, stationary_point='TS')

            # hess_update_freq: Frequency for numerical reevaluation of hessian. = 0 evaluates the numerical hessian in the first iteration and is updated with an BFGS rule, unless approaching a trap region, where it is reevaluated. (Default: 0)

            # numhess_method: Method for evaluating numerical hessian. Forward and central differences are available with “forward” and “central”, respectively. (Default: “forward”)

            # max_iter: Maximum number of optimization steps. (Default: 100)

            # step: Maximum norm between two optimization steps. (Default: 0.1)

            # hmin: Minimum distance between current geometry and stationary point of the quadratic form to consider convergence reached. (Default: 1e-6)

            # gthres: Gradient norm to consider convergence. (Default: 1e-5)

            #self.mol_eq = optimizer.kernel(hess_update_freq=0, numhess_method='forward', max_iter=100, step=0.1, hmin=1e-6, gthres=1e-5)
            
            ####
            # The initial and maximum trust radii may be adjusted by passing (for example) --trust 0.02 --tmax 0.06 on the command line,
            # which is double the default values of 0.01 and 0.03 respectively. Increasing these parameters beyond 0.03 and 0.10 respectively is not recommended.
            prefix = os.path.join(os.path.dirname(os.path.abspath(self.input_file)),'output_opt_ts')

            # Additional options for geomeTRIC
            geometric_options = {
                'prefix': prefix,
                'transition': True, 'trust': 0.02, 'tmax': 0.06}
            self.mol_eq = mf.Gradients().optimizer(solver='geomeTRIC').kernel(geometric_options)
            logging.info('\n Optimized geometry:')
            logging.info('\n')
            if self.unit == 'Bohr':
                logging.info(self.convert_to_custom_xyz(self.mol_eq.atom_coords(unit='Bohr')))
            else:
                logging.info(self.convert_to_custom_xyz(self.mol_eq.atom_coords(unit='ANG')))
            logging.info('\nSCF energy of {0} is {1}.'.format(self.method, self.mol_eq.scf()))
            logging.info('\n')
            logging.info('PySCF optimization complete.')

        
        if (self.job_type == 'opt' or self.job_type == 'conformers') and not self.is_ts:
            from pyscf.geomopt.geometric_solver import optimize
            #from pyscf.geomopt.berny_solver import optimize
            prefix = os.path.join(os.path.dirname(os.path.abspath(self.input_file)),'output_opt')
            conv_params = { # These are the default settings
                                'prefix': prefix,
                                'convergence_energy': 1e-6,  # Eh
                                'convergence_grms': 3e-4,    # Eh/Bohr
                                'convergence_gmax': 4.5e-4,  # Eh/Bohr
                                'convergence_drms': 1.2e-3,  # Angstrom
                                'convergence_dmax': 1.8e-3,  # Angstrom
                            }

            #g_scan = mf.Gradients().as_scanner()
            #self.mol_eq = g_scan.optimizer(solver='geomeTRIC').run()
            #self.mol_q.converged
            self.mol_eq = optimize(mf,maxsteps=self.maxsteps, **conv_params)
            #logging.info(self.mol_eq.atom)
            logging.info('\n Optimized geometry:')
            if self.unit == 'Bohr':
                logging.info(self.convert_to_custom_xyz(self.mol_eq.atom_coords(unit='Bohr')))
            else:
                logging.info(self.convert_to_custom_xyz(self.mol_eq.atom_coords(unit='ANG')))
            logging.info('\nSCF energy of {0} is {1}.'.format(self.method, self.mol_eq.scf()))
            logging.info('\n')
            logging.info('PySCF optimization complete.')

        if self.job_type == 'freq':
            #pip install git+https://github.com/pyscf/properties
            from pyscf.prop.freq import rks, uks
            self.mol_eq = mf.run()
            if DFT:
                if self.scf_type == 'R':
                    w, modes = rks.Freq(self.mol_eq).kernel()
                elif self.scf_type == 'U':
                    w, modes = uks.Freq(self.mol_eq).kernel()
            # print('Frequencies (cm-1):')
            # logging.info('*********************************************')
            # logging.info('Frequencies (cm-1):')
            # print(w)
            # logging.info(w)
            # logging.info('*********************************************')
            # print('Modes:')
            # logging.info('Modes:')
            # logging.info('*********************************************')
            # print(modes)
            # logging.info(modes)
            # logging.info('*********************************************')
            
            # logging.info('Freuqency calculation complete.')
            
            self.output_to_yaml(w, modes)
            try:
                open(os.path.join(os.path.dirname(os.path.abspath(self.input_file)),'output_freq.yml'), 'a').close()
            except:
                pass
        
        # if self.job_type == 'scan':
        #     #https://github.com/pyscf/pyscf/blob/14d88828cd1f18f1e5358da1445355bde55322a1/examples/scf/30-scan_pes.py#L16
        #     # Requires manual development for the code. Not feasible at the moment.
        #     scan_results = []
        #     for value in np.arange(self.start, self.end, self.step_size):
        #         upd
        if self.job_type == 'sp':
            energy = mf.kernel()
            logging.info('The single-point DFT energy is:', energy)
            

# # Run the script
if __name__ == '__main__':
    args = parse_command_line_arguments()
    input_file = args.yml_path  # Directly use the provided path
    input_dir = os.path.dirname(os.path.abspath(input_file))

    script = PYSCFScript_VB(input_file)  # Initialize the script with the YAML file path
    script.run()

#input_file = '/home/calvin/Code/ARC/arc/testing/test_PYSCFAdapter/calcs/Species/EtOH/opt_a370/input.yml'
# output_file = '/home/calvin/Code/ARC/arc/testing/test_PYSCFAdapter/calcs/Species/EtOH/opt_a370/output.log'

# # # # input_file = '/home/calvin/Code/ARC/arc/testing/test_PYSCFAdapter/calcs/TSs/EtOH_ts/opt_a370/input.yml'
# # # # output_file = '/home/calvin/Code/ARC/arc/testing/test_PYSCFAdapter/calcs/TSs/EtOH_ts/opt_a370/output.log'

# # # # input_file = '/home/calvin/Code/ARC/arc/testing/test_PYSCFAdapter/calcs/Species/EtOH_ts/freq_a370/input.yml'
# # # # output_file = '/home/calvin/Code/ARC/arc/testing/test_PYSCFAdapter/calcs/Species/EtOH_ts/freq_a370/output.log'

# input_file = '/home/calvin/Code/PhD/Topic_One/rxns/rxn_0/calcs/Species/rxn_0_[CH2]CCCO/conformer6/input.yml'

# script = PYSCFScript_VB(input_file)
# script.run()
