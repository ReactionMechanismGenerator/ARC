#!/usr/bin/env python3
# encoding: utf-8

"""
A standalone script to run PySCF with the xyz data outputted from ARC.
Should be run under the pyscf_env.
"""

# Parse the input.yml file

import os
import sys
import yaml
import argparse
import pyscf
from pyscf import gto, scf, dft

class PYSCFScript:
    
    def __init__(self, input_file):
        self.input_file = input_file
        self.input_dict = self.read_input_file()
        self.job_type = self.get_job_type()
        self.mol = self.get_mol()
        self.method = self.get_method()
        self.basis = self.get_basis()
        self.charge = self.get_charge()
        self.spin = self.get_spin()
        self.restricted = self.get_restricted()
        self.is_ts = self.get_ts_status()
        self.run()
    def read_input_file(self):
        with open(self.input_file, 'r') as f:
            input_dict = yaml.safe_load(f, Loader=yaml.FullLoader)
        return input_dict
    def get_job_type(self):
        job_type = self.input_dict['job_type']
        return job_type
    def get_mol(self):
        mol = self.input_dict['xyz']
        return mol
    def get_method(self):
        method = self.input_dict['xc_func']
        return method
    def get_basis(self):
        basis = self.input_dict['basis']
        return basis
    def get_charge(self):
        charge = self.input_dict['charge']
        return charge
    def get_spin(self):
        spin = self.input_dict['spin']
        return spin
    def get_restricted(self):
        restricted = self.input_dict['restricted']
        return restricted
    def get_ts_status(self):
        is_ts = self.input_dict['is_ts']
        return is_ts
    def run(self):

        mol = gto.Mole()
        mol.atom = self.mol
        mol.basis = self.basis
        mol.charge = self.charge
        mol.spin = self.spin # TODO: Is this correct [multiplicity -> spin]?
        mol.build()
        
        if self.restricted:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = self.method
        
        # TODO: Check if TS
        if self.is_ts and self.job_type == 'opt':
            from pyscf.qsdopt.qsd_optimizer import QSD
            optimizer = QSD(mf, stationary_point='TS')
            

            # hess_update_freq: Frequency for numerical reevaluation of hessian. = 0 evaluates the numerical hessian in the first iteration and is updated with an BFGS rule, unless approaching a trap region, where it is reevaluated. (Default: 0)

            # numhess_method: Method for evaluating numerical hessian. Forward and central differences are available with “forward” and “central”, respectively. (Default: “forward”)

            # max_iter: Maximum number of optimization steps. (Default: 100)

            # step: Maximum norm between two optimization steps. (Default: 0.1)

            # hmin: Minimum distance between current geometry and stationary point of the quadratic form to consider convergence reached. (Default: 1e-6)

            # gthres: Gradient norm to consider convergence. (Default: 1e-5)

            optimizer.kernel(hess_update_freq=0, numhess_method='forward', max_iter=100, step=0.1, hmin=1e-6, gthres=1e-5)
        
        if self.job_type == 'opt' and not self.is_ts:
            from pyscf.geomopt.geometric_solver import optimize
            conv_params = { # These are the default settings
                                'convergence_energy': 1e-6,  # Eh
                                'convergence_grms': 3e-4,    # Eh/Bohr
                                'convergence_gmax': 4.5e-4,  # Eh/Bohr
                                'convergence_drms': 1.2e-3,  # Angstrom
                                'convergence_dmax': 1.8e-3,  # Angstrom
                            }
            mol_eq = optimize(mf, maxsteps=100, **conv_params) # TODO: Need to define maxsteps as a TRSH parameter
            mf.kernel()
            print('SCF energy of {0} is {1}.'.format(self.method, mf.e_tot))
            
        if self.job_type == 'freq':
            #npip install git+https://github.com/pyscf/properties
            from pyscf.prop.freq import rks            
            w, modes = rks.Freq(mf).kernel()            
            print('Frequencies (cm-1):')
        
        # if self.job_type == 'scan':
        #     #https://github.com/pyscf/pyscf/blob/14d88828cd1f18f1e5358da1445355bde55322a1/examples/scf/30-scan_pes.py#L16
        #     # Requires manual development for the code. Not feasible at the moment.
        #     scan_results = []
        #     for value in np.arange(self.start, self.end, self.step_size):
        #         upd
        if self.job_type == 'sp':
            mf.kernel()
            print('The single-point DFT energy is:', energy)
            
        print('SCF energy of {0} is {1}.'.format(self.method, mf.e_tot))
        
        