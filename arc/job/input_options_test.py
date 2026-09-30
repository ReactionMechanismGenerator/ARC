#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.input_options module
"""

import unittest

from arc.job.input_options import (gaussian_route_sections,
                                   get_input_options,
                                   orca_input_sections,
                                   set_recorded_input_options,
                                   )


class TestInputOptions(unittest.TestCase):
    """
    Contains unit tests of reading the options an ESS input deck was written with, and of recording them per job.
    """

    def test_gaussian_route_sections(self):
        """A deck's route is one line joined by spaces, a log's wrapped route is rejoined as written"""
        deck = '%chk=check.chk\n%mem=1000mb\n#P opt=(calcfc)\n wb97xd/def2tzvp  scf=(tight)\n\ntitle\n\n0 1\nO 0 0 0\n'
        self.assertEqual(gaussian_route_sections(deck.splitlines()),
                         ['#P opt=(calcfc) wb97xd/def2tzvp  scf=(tight)'])
        log = [' ------', ' #P opt=(calcfc) wb97xd/def2tzvp integral=(grid=ultr', ' afine)', ' ------', ' title',
               ' ------', ' #P freq wb97xd/def2tzvp', ' ------']
        self.assertEqual(gaussian_route_sections(log),
                         ['#P opt=(calcfc) wb97xd/def2tzvp integral=(grid=ultrafine)', '#P freq wb97xd/def2tzvp'])
        self.assertEqual(gaussian_route_sections(log, first_only=True),
                         ['#P opt=(calcfc) wb97xd/def2tzvp integral=(grid=ultrafine)'])
        self.assertEqual(gaussian_route_sections(['no route here']), [])

    def test_orca_input_sections(self):
        """Keyword lines, $ lines and % blocks are split in order; the coordinate input is left out"""
        lines = ['!uHF dlpno-ccsd(t) cc-pvtz', '!sp', '%maxcore 1000', '%pal nprocs 8 end', '',
                 '* xyz 0 1', 'O 0 0 0', '*', '', '%scf', 'MaxIter 999', 'end', '', '$new_job', '%coords', 'end']
        self.assertEqual(orca_input_sections(lines),
                         [('!', ['!uHF dlpno-ccsd(t) cc-pvtz']), ('!', ['!sp']), ('maxcore', ['%maxcore 1000']),
                          ('pal', ['%pal nprocs 8 end']), ('scf', ['%scf', 'MaxIter 999', 'end']),
                          ('$', ['$new_job']), ('coords', ['%coords', 'end'])])

    def test_get_input_options_per_program(self):
        """Each program's options are read from its deck as written, without geometry and resource input"""
        orca = '!uHF dlpno-ccsd(t) cc-pvtz cc-pvtz/c tightpno\n!sp \n%maxcore 1000\n%pal nprocs 8 end\n\n' \
               '* xyz 0 2\nO 0 0 0\nH 0 0 1\n*\n\n%scf\nMaxIter 999\nend\n'
        self.assertEqual(get_input_options('orca', orca),
                         '!uHF dlpno-ccsd(t) cc-pvtz cc-pvtz/c tightpno\n!sp\n%scf\nMaxIter 999\nend')
        qchem = '$molecule\n0 1\nO 0 0 0\n$end\n$rem\n   JOBTYPE  opt\n   METHOD   b3lyp\n$end\n\n'
        self.assertEqual(get_input_options('qchem', qchem), '$rem\nJOBTYPE  opt\nMETHOD   b3lyp\n$end')
        molpro = '***,H2O\nmemory,Total=100,m;\n\ngeometry={angstrom;\nO 0 0 0\nH 0 0 1}\n\nbasis=cc-pvtz\n' \
                 'int;\n{hf;\n maxit,999;\n}\nccsd(t);\n---;\n'
        self.assertEqual(get_input_options('molpro', molpro),
                         'basis=cc-pvtz\nint;\n{hf;\nmaxit,999;\n}\nccsd(t);')
        cfour = 'H2O\nO\nH 1 R\n\nR=0.96\n\n*CFOUR(CALC=CCSD,BASIS=PVTZ\nMEMORY_SIZE=100)\n\n'
        self.assertEqual(get_input_options('cfour', cfour), '*CFOUR(CALC=CCSD,BASIS=PVTZ\n)')
        self.assertIsNone(get_input_options('cfour', 'H2O\nO\n'))
        terachem = 'chkfile None\njobname output\nrun minimize\ncoordinates coords.xyz\ngpumem 100\n\n' \
                   'basis def2-svp\nmethod b3lyp\nscrdir scr\n\nend\n$constraint_freeze\nbond 1_2\n$end\n'
        self.assertEqual(get_input_options('terachem', terachem),
                         'run minimize\nbasis def2-svp\nmethod b3lyp\n$constraint_freeze\nbond 1_2\n$end')
        xtb = '/opt/xtb/bin/xtb mol.sdf --opt tight --gfn2 --input input.in --parallel --uhf 0 --chrg 0 ' \
              '> output.out\n' \
              '$constrain\n   force constant=0.05\n$end'
        self.assertEqual(get_input_options('xtb', xtb),
                         '--opt tight --gfn2 --input input.in --uhf 0 --chrg 0\n$constrain\nforce constant=0.05\n$end')
        self.assertEqual(get_input_options('gaussian', '%mem=1mb\n#P sp b3lyp/6-31g\n\ntitle\n'),
                         '#P sp b3lyp/6-31g')
        for software, text in (('psi4', 'molecule {\n}'), ('pyscf', 'job_type: sp'), (None, '#P sp'),
                               ('gaussian', ''), ('gaussian', None), ('qchem', '$molecule\n0 1\n$end\n')):
            self.assertIsNone(get_input_options(software, text), msg=software)

    def test_gaussian_options_keep_every_section_after_the_molecule(self):
        """The route is followed by the sections after the geometry; Link 0, title and molecule are left out"""
        deck = '%chk=check.chk\n%mem=1000mb\n%NProcShared=8\n\n#P opt=(modredundant) b3lyp/genecp  \n scrf=(read)\n\n' \
               'title line\n\n0 1\nI 0 0 0\nH 0 0 1.6\n\nD 1 2 3 4 F\n\nI 0\nLANL2DZ\n****\nH 0\n6-31G(d)\n****\n\n' \
               'I 0\nLANL2DZ\n\nradii=bondi\n\n\n--link1--\n%chk=check.chk\n#P freq b3lyp/genecp\n\nsecond\n\n0 1\n'
        self.assertEqual(get_input_options('gaussian', deck),
                         '#P opt=(modredundant) b3lyp/genecp scrf=(read)\n\nD 1 2 3 4 F\n\n'
                         'I 0\nLANL2DZ\n****\nH 0\n6-31G(d)\n****\n\nI 0\nLANL2DZ\n\nradii=bondi')
        allcheck = '%oldchk=old.chk\n#P sp geom=allcheck guess=read b3lyp/gen\n\nO 0\ncc-pvtz\n****\n'
        self.assertEqual(get_input_options('gaussian', allcheck),
                         '#P sp geom=allcheck guess=read b3lyp/gen\n\nO 0\ncc-pvtz\n****')
        self.assertIsNone(get_input_options('gaussian', '%mem=1mb\ntitle only\n'))

    def test_memory_and_parallelism_settings_are_left_out(self):
        """No program's options state the memory or the cores it was given"""
        orca = '!uHF b3lyp def2-svp PAL8 tightscf\n!pal4\n%maxcore 1000\n%pal nprocs 8 end\n%scf\nMaxIter 999\nend\n'
        self.assertEqual(get_input_options('orca', orca), '!uHF b3lyp def2-svp tightscf\n%scf\nMaxIter 999\nend')
        qchem = '$rem\n   METHOD     b3lyp\n   MEM_TOTAL  4000\n   mem_static=200\n   CC_MEMORY  2000\n' \
                '   THREADS    4\n   BASIS      def2-svp\n$end\n'
        self.assertEqual(get_input_options('qchem', qchem), '$rem\nMETHOD     b3lyp\nBASIS      def2-svp\n$end')
        cfour = 'H2O\nO\n\n*CFOUR(MEMORY_SIZE=100,MEM_UNIT=GB,CALC=CCSD,\nBASIS=PVTZ,MEMORY_SIZE=100\n)\n'
        self.assertEqual(get_input_options('cfour', cfour), '*CFOUR(CALC=CCSD,\nBASIS=PVTZ\n)')
        terachem = 'run energy\ngpus 2 0 1\ngpumem 100\nmethod b3lyp\n'
        self.assertEqual(get_input_options('terachem', terachem), 'run energy\nmethod b3lyp')
        for command in ('xtb mol.sdf --opt --parallel 8 --gfn2 > out', 'xtb mol.sdf --opt -P 8 --gfn2 > out',
                        'xtb mol.sdf --opt --parallel --gfn2 > out', 'xtb mol.sdf --opt --gfn2 --parallel > out'):
            self.assertEqual(get_input_options('xtb', command), '--opt --gfn2', msg=command)
        molpro = '***,H2O\nmemory,Total=100,m;\nbasis=cc-pvtz\nhf;\n'
        self.assertEqual(get_input_options('molpro', molpro), 'basis=cc-pvtz\nhf;')

    def test_cfour_memory_keywords_leave_a_well_formed_keyword_list(self):
        """Removing MEMORY_SIZE and MEM_UNIT keeps every remaining keyword and separator in the deck's layout"""
        cases = (('*CFOUR(CALC=CCSD,MULTIPLICITY=1,MEMORY_SIZE=224000000\nFROZEN_CORE=ON,SCF_CONV=10)',
                  '*CFOUR(CALC=CCSD,MULTIPLICITY=1\nFROZEN_CORE=ON,SCF_CONV=10)'),
                 ('*CFOUR(CALC=CCSD,MULTIPLICITY=1,MEMORY_SIZE=224000000\n)',
                  '*CFOUR(CALC=CCSD,MULTIPLICITY=1\n)'),
                 ('*CFOUR(CALC=CCSD,MULTIPLICITY=1,MEMORY_SIZE=2,MEM_UNIT=GB\nFROZEN_CORE=ON)',
                  '*CFOUR(CALC=CCSD,MULTIPLICITY=1\nFROZEN_CORE=ON)'),
                 ('*CFOUR(CALC=CCSD,MULTIPLICITY=1 , MEMORY_SIZE = 2 , MEM_UNIT = GB\nFROZEN_CORE=ON)',
                  '*CFOUR(CALC=CCSD,MULTIPLICITY=1\nFROZEN_CORE=ON)'),
                 ('*CFOUR(MEMORY_SIZE = 2 , MEM_UNIT = GB,CALC=CCSD\nBASIS=PVTZ)', '*CFOUR(CALC=CCSD\nBASIS=PVTZ)'),
                 ('*CFOUR(CALC=CCSD,MEMORY_SIZE=2,BASIS=PVTZ)', '*CFOUR(CALC=CCSD,BASIS=PVTZ)'),
                 ('*CFOUR(CALC=CCSD\nMEMORY_SIZE=2,BASIS=PVTZ)', '*CFOUR(CALC=CCSD\nBASIS=PVTZ)'),
                 ('*CFOUR(CALC=CCSD,BASIS=PVTZ)', '*CFOUR(CALC=CCSD,BASIS=PVTZ)'))
        for keyword_list, expected in cases:
            with self.subTest(keyword_list=keyword_list):
                options = get_input_options('cfour', f'H2O\nO\nH 1 R\n\nR=0.96\n\n{keyword_list}\n\n')
                self.assertEqual(options, expected)
                self.assertNotRegex(options, r',\s*\)|,\s*\n|\(\s*,|,,')

    def test_gaussian_geom_allcheck_is_read_from_the_route_options(self):
        """Both allcheck spellings, in any geom form, mean no title or molecule; a comment does not"""
        basis = '\n\nO 0\ncc-pvtz\n****\n'
        for route in ('#P sp geom=allcheck guess=read b3lyp/gen', '#P sp geom=allchk guess=read b3lyp/gen',
                      '#P sp Geom=(AllChk,NewDefinition) guess=read b3lyp/gen', '#P sp geom(allcheck) b3lyp/gen',
                      '#P sp\n geometry = ( newdefinition, allchk ) b3lyp/gen'):
            with self.subTest(route=route):
                options = get_input_options('gaussian', f'%oldchk=old.chk\n{route}{basis}')
                self.assertTrue(options.endswith('\n\nO 0\ncc-pvtz\n****'), msg=options)
        deck = '#P sp b3lyp/gen ! not geom=allchk, see geom=(allcheck)\n\ntitle\n\n0 1\nO 0 0 0\n\nO 0\ncc-pvtz\n****\n'
        self.assertEqual(get_input_options('gaussian', deck),
                         '#P sp b3lyp/gen ! not geom=allchk, see geom=(allcheck)\n\nO 0\ncc-pvtz\n****')
        for route in ('#P sp geom=check guess=read b3lyp/gen', '#P sp geom=(check,modredundant) b3lyp/gen'):
            with self.subTest(route=route):
                self.assertEqual(get_input_options('gaussian', f'{route}\n\ntitle\n\n0 1\n\nO 0\ncc-pvtz\n****\n'),
                                 f'{route}\n\nO 0\ncc-pvtz\n****')

    def test_set_recorded_input_options(self):
        """A job's options are recorded under its job key and removed by None or an empty string"""
        entry = dict()
        set_recorded_input_options(entry, 'opt', '#P opt b3lyp/6-31g')
        self.assertEqual(entry, {'input_options': {'opt': '#P opt b3lyp/6-31g'}})
        set_recorded_input_options(entry, 'opt', None)
        self.assertEqual(entry['input_options'], dict())
        set_recorded_input_options(entry, 'sp', '')
        self.assertEqual(entry['input_options'], dict())


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
