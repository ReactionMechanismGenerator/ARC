#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics_qa module
"""

import copy
import itertools
import json
import os
import shutil
import tempfile
import subprocess
import unittest
from unittest import mock

from arc.common import get_angle_in_180_range
import numpy as np
from rdkit import Chem

from arc.family import get_reaction_family_products
from arc.imports import settings
from arc.job.adapters.ts.heuristics import HeuristicsAdapter, get_multiple_bond_pairs, load_family_parameters
from arc.job.adapters.ts.heuristics_qa import (check_sn2_ts_bonds,
                                               get_oh_d1_atom_order,
                                               filter_ts_guesses_with_uma_sp,
                                               remove_duplicate_ts_guesses,
                                               get_ca_cb_staggered_rotamers,
                                               get_stretched_sn2_zmat,
                                               SN2_TS_N_CA_X_ANGLE_RANGE,
                                               is_ring_bond,
                                               qa_sn2,
                                               qa_hofmann,
                                               check_hofmann_ts_bonds,
                                               get_hofmann_anti_periplanar_conformers,
                                               get_unique_hofmann_product_dicts,
                                               is_pi_activated_beta_carbon,
                                               stretch_xyz_bond,
                                               get_stereo_consistent_xyz,
                                               get_rdkit_stereo_smiles,
                                               )
from arc.reaction import ARCReaction
from arc.species.converter import rdkit_conf_from_mol, sort_xyz_using_indices, str_to_xyz, zmat_to_xyz
from arc.species.species import ARCSpecies, colliding_atoms
from arc.species.vectors import calculate_angle, calculate_dihedral_angle, calculate_distance


class TestHeuristicsQA(unittest.TestCase):
    """
    Contains unit tests for the QA TS-guess functions of the heuristics adapter.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.scratch_dir = tempfile.mkdtemp(prefix='arc_test_heuristics_')
        cls.addClassCleanup(shutil.rmtree, cls.scratch_dir, ignore_errors=True)
        cls.h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
                         H      -0.76330345   -0.19953755    0.00000000
                         H       0.76363177   -0.19827735    0.00000000"""
        cls.h2o_xyz = str_to_xyz("""O      -0.00032832    0.39781490    0.00000000
                                    H      -0.76330345   -0.19953755    0.00000000
                                    H       0.76363177   -0.19827735    0.00000000""")

        cls.tma = ARCSpecies(label='TMA', smiles='C[N+](C)(C)C', xyz="""C   0.32629948   1.46357429  -0.26044001
        N  -0.00000001  -0.00000004  -0.00000002
        C  -0.53562668  -0.63505944  -1.27520715
        C  -1.05144842  -0.09811020   1.09598369
        C   1.26077561  -0.73040463   0.43966336
        H  -0.58986326   1.97215483  -0.57490666
        H   1.08232075   1.51488451  -1.04955136
        H   0.70928106   1.90320241   0.66527590
        H   0.23196131  -0.55558259  -2.05070036
        H  -0.76441414  -1.68498288  -1.06974487
        H  -1.44022261  -0.09831400  -1.57605350
        H  -1.94912095   0.43142931   0.76331336
        H  -0.64997572   0.36247710   2.00349483
        H  -1.27331421  -1.15523964   1.26962329
        H   1.63121642  -0.26133127   1.35598324
        H   2.00425486  -0.64964994  -0.35884503
        H   1.00787712  -1.77904796   0.62211151""")
        cls.qa_pyrrolidinium_xyz = """C      -1.50297674    0.19876984   -1.17422303
N      -0.57863163   -0.06210149    0.00514151
C      -1.40571311   -0.51094414    1.20006613
C       0.22405954    1.20958185    0.34852397
C       1.65394749    0.73069811    0.50407400
C       1.76221417   -0.36405918   -0.53442800
C       0.46945917   -1.13515048   -0.35468412
H      -2.18403366    1.01089899   -0.90350864
H      -0.89336006    0.47931640   -2.03834094
H      -2.06186181   -0.71803396   -1.38375322
H      -2.14263833    0.26678715    1.42110884
H      -0.73536750   -0.65411941    2.05281243
H      -1.90358446   -1.44960144    0.93982555
H       0.13415125    1.90568310   -0.49297806
H      -0.18291538    1.67135766    1.25323613
H       2.37775301    1.53445523    0.34253925
H       1.81400925    0.32414545    1.50969160
H       1.81467274    0.06530219   -1.54193409
H       2.64280992   -0.99562362   -0.38714973
H       0.54410049   -1.83288843    0.48695511
H       0.15390565   -1.67447382   -1.25297469"""
        cls.oh_d1 = ARCSpecies(label='OH_d1', smiles='[OH-].O', xyz="""O   0.09218386  -1.12466927   0.00000000
        O  -0.30799026   1.48010372   0.00000000
        H   0.14642271  -2.10188764   0.00000000
        H  -0.56801745   0.50781377   0.00000000
        H   0.63740113   1.23863942   0.00000000""")
        cls.methanol_water = ARCSpecies(label='MeOH_water', smiles='CO.O', xyz="""C   0.18578383  -0.74379564   0.70625041
        O   0.78804603  -1.78016323   1.46352191
        O  -0.69121538   1.68322749  -1.41943841
        H   0.80584717   0.15411066   0.76272470
        H   0.08250090  -1.06654728  -0.33253152
        H  -0.80226276  -0.52985803   1.12035918
        H   0.84185459  -1.45954453   2.37979857
        H  -1.32259038   1.59349278  -2.15004196
        H  -0.70720706   2.63181294  -1.21844482""")
        cls.trimethylamine = ARCSpecies(label='NMe3', smiles='CN(C)C', xyz="""C   1.31438808  -0.44451979   0.02289995
        N   0.03378693   0.02743087  -0.50251848
        C  -1.03831158  -0.90110194  -0.14590539
        C  -0.26252131   1.36795188   0.00141230
        H   1.56377815  -1.42582986  -0.39602755
        H   2.12326729   0.23511663  -0.26774394
        H   1.30947115  -0.52728195   1.11590377
        H  -1.12822746  -0.98477449   0.94293767
        H  -2.00872404  -0.58928786  -0.54914581
        H  -0.81644887  -1.90672625  -0.52028739
        H   0.49732543   2.10022805  -0.29512729
        H  -1.23515940   1.71145197  -0.36824704
        H  -0.32723290   1.36198629   1.09523222""")
        # A quaternary ammonium with a terminal alkyne: the C#C bond is shorter than the
        # collision threshold for a C-C single bond.
        cls.butynyl_tma = ARCSpecies(label='butynyl_TMA', smiles='C[N+](C)(C)CCC#C',
                                     xyz="""C       0.42402100    1.37019800    0.48046300
        N       1.03981000    0.10247600    0.00061200
        C       2.13767200    0.43422200   -0.95323700
        C       1.62400700   -0.63237400    1.15769500
        C       0.04952600   -0.75820900   -0.74360000
        C      -1.22359700   -1.13795300    0.00605800
        C      -2.27434900   -0.12482600    0.00117700
        C      -3.14847700    0.69390200   -0.01540600
        H      -0.04759300    1.87004800   -0.36065300
        H       1.21380800    1.99063900    0.89447900
        H      -0.30695300    1.14563400    1.25003200
        H       2.88373500    1.02298000   -0.42783000
        H       2.57498600   -0.49173900   -1.31604700
        H       1.71990300    1.00420600   -1.77801400
        H       0.84516500   -0.85532600    1.87932600
        H       2.07875700   -1.54880100    0.79263000
        H       2.37435400    0.00446700    1.61762400
        H      -0.19525800   -0.22084100   -1.65686500
        H       0.59275800   -1.66309000   -1.00401700
        H      -1.01181300   -1.44360700    1.03049900
        H      -1.60902200   -2.02632500   -0.49660400
        H      -3.92472600    1.42476300   -0.02707800""")
        cls.butynol_water = ARCSpecies(label='butynol_water', smiles='OCCC#C.O',
                                       xyz="""O       0.68908100    1.53323500   -0.46308800
        C       1.35765900    0.65498900    0.41151700
        C       1.36613200   -0.79871700   -0.07129900
        C       0.05194700   -1.43203000   -0.06670400
        C      -1.02840900   -1.95556800   -0.05419200
        O      -2.03793100    1.11607900    0.20668000
        H      -0.26180800    1.47703900   -0.28783500
        H       2.39434000    0.98723500    0.47082400
        H       0.93452700    0.70388300    1.41961200
        H       2.03285300   -1.37407100    0.57498500
        H       1.77893000   -0.83783900   -1.08106900
        H      -1.97752100   -2.44201000   -0.04640100
        H      -2.68539400    1.31802600   -0.47437800
        H      -1.90762500    0.16100700    0.15718800""")
        # Quaternary ammonium substrates for the qa_hofmann (Hofmann E2 elimination) family.
        cls.ethyl_tma = ARCSpecies(label='ethyl_TMA', smiles='CC[N+](C)(C)C',
                                   xyz="""C       2.04332265   -0.16348784   -0.59037792
        C       1.07050142   -0.07307567    0.57401067
        N      -0.40256562    0.03261813    0.14500682
        C      -0.65588555    1.28943829   -0.67958677
        C      -1.25335040    0.11529406    1.41670566
        C      -0.84440227   -1.19441641   -0.64422628
        H       1.86888394   -1.05040440   -1.20530934
        H       3.06592213   -0.23560500   -0.20430421
        H       2.00339313    0.72183063   -1.23054015
        H       1.15746058   -0.96314852    1.20791261
        H       1.29213396    0.81126632    1.18264999
        H      -0.14193019    1.19475222   -1.63932691
        H      -0.28127654    2.15171943   -0.12064960
        H      -1.73346317    1.38103451   -0.84713996
        H      -1.08403379   -0.79148018    2.00502591
        H      -0.94787930    1.00241834    1.97948977
        H      -2.30590740    0.19106571    1.12755336
        H      -0.60299460   -2.08718119   -0.06030434
        H      -1.92382041   -1.12706985   -0.81143297
        H      -0.32410855   -1.20556857   -1.60515635""")
        # A propyl quaternary ammonium whose conformer has both beta hydrogens gauche to N.
        cls.propyl_tma = ARCSpecies(label='propyl_TMA', smiles='CCC[N+](C)(C)C',
                                    xyz="""C       2.95844893    0.05001432    0.32699272
        C       1.55695221   -0.52618411    0.17490150
        C       0.52361084    0.59333433    0.05489702
        N      -0.92293272    0.08896424   -0.10254277
        C      -1.36638487   -0.72741375    1.10588217
        C      -1.83385863    1.31692843   -0.20955849
        C      -1.09247290   -0.74002527   -1.37039368
        H       3.69267575   -0.75725285    0.41231881
        H       3.03159470    0.67194854    1.22493034
        H       3.22676127    0.66296415   -0.53946293
        H       1.54460152   -1.16733695   -0.71190413
        H       1.35037551   -1.15839410    1.04397560
        H       0.54156689    1.22877234    0.94865698
        H       0.73717072    1.21976544   -0.81968129
        H      -0.80394252   -1.66401085    1.12867075
        H      -2.43436236   -0.94331988    1.00357267
        H      -1.18011513   -0.13886828    2.00873444
        H      -2.86732222    0.97558894   -0.32213588
        H      -1.52744462    1.89903054   -1.08379178
        H      -1.72526685    1.90813920    0.70460429
        H      -2.15723221   -0.95608155   -1.50179416
        H      -0.71273574   -0.16038611   -2.21658670
        H      -0.53968758   -1.67617676   -1.26028547""")
        # A phenethyl quaternary ammonium: the phenyl group activates the beta hydrogens.
        cls.phenethyl_tma = ARCSpecies(label='phenethyl_TMA', smiles='C[N+](C)(C)CCc1ccccc1',
                                       xyz="""C      -2.00718970   -1.52466135   -0.55346043
        N      -1.77997235   -0.24464020    0.23723565
        C      -3.13452730    0.23934602    0.76877763
        C      -1.23357766    0.83703511   -0.69147801
        C      -0.86056010   -0.47819989    1.46039650
        C       0.51031386   -1.11893719    1.17779020
        C       1.52734514   -0.24362687    0.49049006
        C       1.91567056    0.98606616    1.04294677
        C       2.85671788    1.78805893    0.39509111
        C       3.43083285    1.36488933   -0.80104186
        C       3.07477342    0.13364805   -1.34534398
        C       2.13407205   -0.67073755   -0.70046722
        H      -2.34914363   -2.30138242    0.13645244
        H      -2.76678377   -1.32974516   -1.31679957
        H      -1.07087541   -1.81716959   -1.03499984
        H      -3.80200598    0.41316132   -0.08069743
        H      -3.54707282   -0.53575409    1.42152429
        H      -2.97434368    1.16766645    1.32519180
        H      -0.95834520    1.70754331   -0.09001117
        H      -2.01925302    1.10487563   -1.40524345
        H      -0.38017434    0.44130311   -1.24160076
        H      -1.41861752   -1.14804752    2.12662879
        H      -0.75972226    0.48787494    1.96832214
        H       0.38316388   -2.06099809    0.63321059
        H       0.94897230   -1.40804641    2.14278976
        H       1.51133015    1.33068803    1.99224412
        H       3.15971963    2.73751874    0.83189952
        H       4.17407568    1.98502849   -1.29779068
        H       3.54464384   -0.20573304   -2.26618790
        H       1.89053351   -1.63702423   -1.13779383""")
        cls.ethene = ARCSpecies(label='ethene', smiles='C=C')
        cls.propene = ARCSpecies(label='propene', smiles='C=CC')
        cls.styrene = ARCSpecies(label='styrene', smiles='C=Cc1ccccc1')
        cls.water_dimer = ARCSpecies(label='water_dimer', smiles='O.O')
        # A neopentyl quaternary ammonium: Cb is quaternary, so a Cb methyl may block the backside of Ca.
        cls.neopentyl_tma = ARCSpecies(label='neopentyl_TMA', smiles='C[N+](C)(C)CC(C)(C)C',
                                       xyz="""C      -1.41688700   -1.00936900   -1.03515200
        N      -1.33459400   -0.00544300    0.05928700
        C      -2.36928200   -0.34452600    1.08253300
        C      -1.66499300    1.34327700   -0.48414500
        C      -0.00053000   -0.01638500    0.78525600
        C       1.35422000    0.00280000    0.04893800
        C       1.50295100    1.10802100   -0.99586100
        C       1.71356600   -1.35170700   -0.57170200
        C       2.35732200    0.27373500    1.18288000
        H      -2.44665700   -1.04600600   -1.38017000
        H      -1.11842800   -1.97856200   -0.64675200
        H      -0.77130700   -0.70464800   -1.85107800
        H      -2.19010700   -1.35410700    1.44054900
        H      -3.34825900   -0.27994500    0.61651500
        H      -2.29498200    0.36628600    1.90093600
        H      -2.68063300    1.31155400   -0.86888400
        H      -0.98134000    1.59373000   -1.28462600
        H      -1.59194200    2.06955800    0.32015900
        H      -0.05820300    0.84573800    1.44731700
        H      -0.02601500   -0.91516700    1.39882400
        H       1.21971700    2.08235200   -0.59469000
        H       0.91045100    0.91061500   -1.88961700
        H       2.54722600    1.16497100   -1.30767100
        H       1.13556800   -1.57584300   -1.46559700
        H       1.56692800   -2.16280500    0.14402900
        H       2.76590600   -1.34586300   -0.86055900
        H       2.27628200   -0.47654900    1.97209000
        H       3.37394000    0.24061900    0.78997800
        H       2.19579700    1.25708800    1.62775300""")
        cls.neopentanol_water = ARCSpecies(label='neopentanol_water', smiles='OCC(C)(C)C.O',
                                           xyz="""O       1.13050100    1.16290100   -0.08913400
        C      -0.02332900    0.72961800   -0.80574400
        C      -0.99775700   -0.12618600    0.00825700
        C      -1.55771500    0.68066000    1.18113600
        C      -2.13906300   -0.52024500   -0.92982200
        C      -0.29772500   -1.38140900    0.52662800
        O       3.32722200   -0.57806300    0.06112200
        H       0.85323300    1.70969400    0.64989800
        H      -0.55077200    1.60084000   -1.20785100
        H       0.35954600    0.14984700   -1.64687900
        H      -2.03106800    1.60223100    0.83464600
        H      -0.78157500    0.94500100    1.90372100
        H      -2.30803500    0.09822500    1.71898600
        H      -1.76755800   -1.09139400   -1.78354600
        H      -2.86871700   -1.13852200   -0.40375500
        H      -2.65915600    0.36102100   -1.31197100
        H       0.52416500   -1.12957800    1.19817400
        H      -1.00039600   -2.00857800    1.07901200
        H       0.10726300   -1.97280700   -0.29753600
        H       2.55484400    0.01067100    0.04319200
        H       4.00201000   -0.10923200   -0.43327200""")
        # Beta-branched quaternary ammoniums (a ring at Cb), in conformers where a Cb substituent sits in the
        # backside-attack path of Ca.
        cls.cyclopentylmethyl_tma = ARCSpecies(label='cyclopentylmethyl_TMA', smiles='C[N+](C)(C)CC1CCCC1',
                                               xyz="""C       1.76448500    1.43945100    0.34753900
        N       1.82842900   -0.00277800   -0.01061500
        C       2.89156000   -0.19534000   -1.03719300
        C       2.18952700   -0.79535400    1.19680600
        C       0.52538500   -0.49157700   -0.60012700
        C      -0.72921600   -0.25165500    0.22380300
        C      -1.39018100    1.14451500    0.07298400
        C      -2.87932900    0.88589000   -0.22423300
        C      -3.11777400   -0.56019100    0.20852000
        C      -1.82185800   -1.24897500   -0.19629600
        H       1.45977900    2.00367500   -0.52906300
        H       1.05447800    1.57677200    1.15629900
        H       2.75391100    1.75196400    0.66983000
        H       3.84277300    0.11202000   -0.61182200
        H       2.65306300    0.41375000   -1.90447400
        H       2.92424300   -1.24609400   -1.31136700
        H       1.43812300   -0.64826800    1.96509400
        H       2.24653300   -1.84392200    0.91877200
        H       3.15379100   -0.44783400    1.55644200
        H       0.67608500   -1.55831700   -0.75779000
        H       0.43388700   -0.00873400   -1.57216300
        H      -0.51401900   -0.42943900    1.27754500
        H      -1.27151300    1.72018700    0.99033100
        H      -0.92400200    1.72039100   -0.72625700
        H      -3.06552600    0.97466900   -1.29641100
        H      -3.53395600    1.59602200    0.27987200
        H      -3.24258100   -0.61456600    1.29317800
        H      -3.99990900   -1.00385700   -0.25243300
        H      -1.67737200   -2.22763700    0.26093000
        H      -1.80252700   -1.38179700   -1.28245600""")
        cls.cyclopentylmethanol_water = ARCSpecies(label='cyclopentylmethanol_water', smiles='OCC1CCCC1.O',
                                                   xyz="""O      -1.87443100   -0.85577500    0.94312700
        C      -0.78249200   -1.26301800    0.13950900
        C       0.18217300   -0.13985400   -0.19978600
        C       1.36680600   -0.62866100   -1.06915200
        C       2.63918800   -0.01120900   -0.45415100
        C       2.13145600    1.10393800    0.46159300
        C       0.83299700    0.52724300    1.01727300
        O      -3.41002500    0.95374400   -0.67063800
        H      -2.40407500   -0.23059800    0.42842300
        H      -1.14200700   -1.72907200   -0.78676600
        H      -0.25161800   -2.03376800    0.70510400
        H      -0.36914000    0.62510800   -0.75535800
        H       1.23571200   -0.31767600   -2.10541300
        H       1.42297800   -1.71926500   -1.07168000
        H       3.33853000    0.34585400   -1.20998700
        H       3.16334600   -0.75840900    0.14524500
        H       1.91460400    2.00185300   -0.12371900
        H       2.84817500    1.37883400    1.23600500
        H       0.19101800    1.27403000    1.48329200
        H       1.06112400   -0.23194900    1.77268400
        H      -2.83629000    1.61530600   -1.06749100
        H      -4.08139200    1.45667400   -0.20117400""")
        cls.cyclohexylmethyl_tma = ARCSpecies(label='cyclohexylmethyl_TMA', smiles='C[N+](C)(C)CC1CCCCC1',
                                              xyz="""C       2.10195000    1.37939000    0.60291900
        N       2.13562400    0.00033200    0.04503900
        C       3.31371500   -0.11792700   -0.85994800
        C       2.29026600   -0.98365200    1.15157000
        C       0.89976600   -0.30259500   -0.76815500
        C      -0.43913700   -0.13434500   -0.04717000
        C      -1.13235300    1.18076200   -0.41556400
        C      -2.48398400    1.31727800    0.27718800
        C      -3.39023000    0.13090200   -0.03526700
        C      -2.71009500   -1.18794300    0.31847800
        C      -1.35335200   -1.32157700   -0.36464900
        H       1.97016700    2.08099300   -0.21579600
        H       3.04651200    1.56378000    1.10721400
        H       1.28538100    1.46291200    1.31284200
        H       4.21387300    0.07038000   -0.28168100
        H       3.33559100   -1.12231600   -1.27314600
        H       3.21674900    0.61622000   -1.65477400
        H       1.45543600   -0.89291200    1.83841600
        H       2.32107600   -1.98187500    0.72435100
        H       3.21894100   -0.76682100    1.67223200
        H       1.03513800   -1.32937100   -1.10374900
        H       0.95262300    0.34934100   -1.63879100
        H      -0.27769000   -0.13752800    1.03374500
        H      -1.27737000    1.20210700   -1.50092100
        H      -0.49504800    2.03021300   -0.16574500
        H      -2.96252000    2.25046300   -0.02525400
        H      -2.32784100    1.37693600    1.35927300
        H      -3.63095400    0.13588100   -1.10329800
        H      -4.33485900    0.22389300    0.50381100
        H      -2.56832300   -1.23854100    1.40293300
        H      -3.34766600   -2.02993200    0.04300300
        H      -0.86416700   -2.24835700   -0.05679700
        H      -1.49368600   -1.37954900   -1.44950800""")
        cls.cyclohexylmethanol_water = ARCSpecies(label='cyclohexylmethanol_water', smiles='OCC1CCCCC1.O',
                                                  xyz="""O      -2.44876700   -0.76816400   -0.43571500
        C      -1.23435600   -1.49745300   -0.24600300
        C      -0.12323600   -0.54111400    0.13626000
        C       0.15442400    0.49535600   -0.95351400
        C       1.28890300    1.43533200   -0.55729600
        C       2.55841600    0.66153100   -0.21465400
        C       2.29279200   -0.38540100    0.86326000
        C       1.15118500   -1.31819600    0.46898900
        O      -2.80165400    1.72473400    0.85546900
        H      -3.15941500   -1.38776400   -0.61203300
        H      -1.37018600   -2.24247300    0.54349100
        H      -0.97289300   -2.02325700   -1.17084800
        H      -0.43866400   -0.01177300    1.04360700
        H      -0.74853700    1.06711600   -1.17369900
        H       0.42528200   -0.03299500   -1.87494100
        H       0.97929100    2.02262400    0.31385700
        H       1.48410600    2.14555100   -1.36312000
        H       3.34361800    1.34679500    0.11085700
        H       2.92642000    0.16171600   -1.11679700
        H       3.19693000   -0.96520200    1.05854700
        H       2.03290700    0.12101200    1.79849300
        H       0.95313100   -2.03045800    1.27326600
        H       1.44892500   -1.90286700   -0.40862600
        H      -2.69798200    0.86028100    0.42653800
        H      -1.92833900    2.11935300    0.80974900""")
        # Ca alpha to an amide carbonyl (sp2 Cb): rigidly rotating the carbonyl O anti to N swings the NMe2
        # group into the NMe3 group.
        cls.amide_alpha_tma = ARCSpecies(label='amide_alpha_TMA', smiles='C[N+](C)(C)CC(=O)N(C)C',
                                         xyz="""C       2.02738700   -0.77679300   -1.16988100
        N       1.70857300    0.08455600    0.00756400
        C       2.64270700    1.25123700   -0.00233800
        C       1.92295700   -0.67269600    1.27585000
        C       0.32352500    0.64423800   -0.07431300
        C      -0.81335500   -0.37380200   -0.04893200
        O      -0.61073400   -1.58274000   -0.02684900
        N      -2.04151300    0.17168200   -0.06002800
        C      -3.21063700   -0.68052400    0.05075700
        C      -2.29830300    1.60131200   -0.00023400
        H       3.06926800   -1.07363900   -1.09225600
        H       1.37984900   -1.64433500   -1.15686300
        H       1.86996300   -0.19209200   -2.07235700
        H       2.49377100    1.80999200   -0.92205300
        H       2.43097500    1.87638300    0.86038800
        H       3.65897500    0.87188600    0.04795800
        H       2.97149800   -0.95158400    1.32494300
        H       1.29361800   -1.55373500    1.27119100
        H       1.66859200   -0.02273500    2.10866100
        H       0.27821000    1.21867500   -0.99748200
        H       0.21905000    1.33367400    0.76108500
        H      -2.90708000   -1.72141100    0.04029800
        H      -3.88561300   -0.49119400   -0.78443100
        H      -3.73877700   -0.46863000    0.98246400
        H      -1.58019600    2.16808700   -0.58637700
        H      -2.28704400    1.96380400    1.03003400
        H      -3.28398800    1.78797500   -0.42095300""")

    def test_check_sn2_ts_bonds(self):
        """
        Test check_sn2_ts_bonds with passing and failing geometries.
        """
        # Atom layout used throughout:
        #   0=Ca, 1=N, 2=R1(H), 3=O_r, 4=H_r, 5=H1_w, 6=O_w, 7=H2_w
        # atom_indices = [O_r_seq=3, H_r_seq=4, Ca_xyz=0, H1_w_seq=5, O_w_seq=6]
        INDICES = [3, 4, 0, 5, 6]

        def make_xyz(coords):
            return {'symbols': ('C', 'N', 'H', 'O', 'H', 'H', 'O', 'H'),
                    'isotopes': (12, 14, 1, 16, 1, 1, 16, 1),
                    'coords': coords}

        valid_coords = (
            ( 0.0,   0.0,  0.0),
            ( 1.5,   0.0,  0.0),
            ( 0.0,   1.0,  0.0),
            (-2.07,  0.0,  0.0),
            (-2.07,  0.96, 0.0),
            (-4.69,  0.0,  0.0),
            (-5.696, 0.0,  0.0),
            (-5.696, 0.958,0.0),
        )
        self.assertTrue(check_sn2_ts_bonds(make_xyz(valid_coords), INDICES))

        # Failing: heavy atom N closer to O_r than Ca
        fail_heavy = list(valid_coords)
        fail_heavy[1] = (2.5, 0.0, 0.0)   # N far right
        fail_heavy[3] = (1.5, 0.0, 0.0)   # O_r at 1.5 from Ca, 1.0 from N → N is first heavy atom
        fail_heavy[4] = (1.5, 0.96, 0.0)
        fail_heavy[5] = (-1.12, 0.0, 0.0)
        fail_heavy[6] = (-2.126, 0.0, 0.0)
        self.assertFalse(check_sn2_ts_bonds(make_xyz(tuple(fail_heavy)), INDICES))

        # Failing: H_r not bonded to O_r (Ca is closer to H_r)
        fail_hr = list(valid_coords)
        fail_hr[4] = (-1.0, 0.0, 0.0)   # H_r: 1.0 Å from Ca, 1.07 Å from O_r
        self.assertFalse(check_sn2_ts_bonds(make_xyz(tuple(fail_hr)), INDICES))

        # Failing: H1_w nearest is not O_w (H1_w collapsed onto O_r)
        fail_h1w_nearest = list(valid_coords)
        fail_h1w_nearest[5] = (-3.0, 0.0, 0.0)   # H1_w: 0.93 Å from O_r, 2.696 Å from O_w → O_r is nearest
        self.assertFalse(check_sn2_ts_bonds(make_xyz(tuple(fail_h1w_nearest)), INDICES))

        # Failing: H1_w nearest is O_w but second nearest is not O_r (H-bond broken)
        fail_h1w_second = list(valid_coords)
        fail_h1w_second[3] = (-8.0, 0.0, 0.0)   # O_r moved far away → second nearest to H1_w is no longer O_r
        fail_h1w_second[4] = (-8.0, 0.96, 0.0)  # H_r follows O_r
        self.assertFalse(check_sn2_ts_bonds(make_xyz(tuple(fail_h1w_second)), INDICES))

        # Failing: H2_w detached from O_w
        fail_h2w = list(valid_coords)
        fail_h2w[7] = (-2.07, 0.0, 0.0)   # H2_w lands on O_r position → nearest is O_r not O_w
        self.assertFalse(check_sn2_ts_bonds(make_xyz(tuple(fail_h2w)), INDICES))

        # Failing: O_w's two nearest atoms are not both water H's (H1_w and H2_w far away)
        fail_ow = list(valid_coords)
        fail_ow[5] = (-4.69, 5.0, 0.0)   # H1_w moved far from O_w
        fail_ow[7] = (-5.696, 5.0, 0.0)  # H2_w moved far from O_w
        self.assertFalse(check_sn2_ts_bonds(make_xyz(tuple(fail_ow)), INDICES))

    def test_heuristics_for_qa_sn2(self):
        """
        Test heuristics for QA SN2: C[N+](C)(C)C + [OH-].O <=> CO.O + CN(C)C.
        """
        rxn = ARCReaction(r_species=[self.tma, self.oh_d1],
                          p_species=[self.methanol_water, self.trimethylamine],
                          family='qa_sn2')
        adapter = HeuristicsAdapter(job_type='tsg',
                                    reactions=[rxn],
                                    testing=True,
                                    project='test',
                                    project_directory=os.path.join(self.scratch_dir, 'heuristics_qasn2'))
        with mock.patch.dict(settings, {'UMA_TS_GUESS_PREFILTER': False}):
            adapter.execute_incore()
        self.assertEqual(rxn.family, 'qa_sn2')
        self.assertTrue(rxn.ts_species.is_ts)
        self.assertGreater(len(rxn.ts_species.ts_guesses), 0)
        symbols = rxn.ts_species.ts_guesses[0].initial_xyz['symbols']
        self.assertEqual(len(symbols), 22)
        self.assertEqual(sorted(symbols[17:]), ['H', 'H', 'H', 'O', 'O'])
        # Reaction centre of every guess: N is atom 1, the attacking O_r (charged O of OH_d1) is atom 17,
        # and Ca is the methyl carbon nearest O_r.
        n, o_r = 1, 17
        r_ca_n = calculate_distance(self.tma.get_xyz()['coords'], [0, n], index=0)
        for tsg in rxn.ts_species.ts_guesses:
            coords = tsg.initial_xyz['coords']
            ca = min([0, 2, 3, 4], key=lambda i: calculate_distance(coords, [i, o_r], index=0))
            self.assertAlmostEqual(calculate_distance(coords, [ca, n], index=0), 1.3 * r_ca_n, delta=0.01)
            self.assertAlmostEqual(calculate_distance(coords, [ca, o_r], index=0), 2.07, delta=0.01)
            self.assertAlmostEqual(calculate_angle(coords, [n, ca, o_r], index=0), 166.0, delta=0.5)

    def test_get_stretched_sn2_zmat_ring(self):
        """
        Test that a ring Ca-N bond is stretched without stretching any other ring bond:
        C[N+]1(C)CCCC1, Ca = ring carbon 3, N = 1.
        """
        qa = ARCSpecies(label='QA', smiles='C[N+]1(C)CCCC1', xyz=self.qa_pyrrolidinium_xyz)
        xyz, mol = qa.get_xyz(), qa.mol
        self.assertTrue(is_ring_bond(mol, 3, 1))
        self.assertFalse(is_ring_bond(mol, 0, 1))
        self.assertFalse(is_ring_bond(mol, 3, 0))  # not bonded
        self.assertFalse(is_ring_bond(None, 3, 1))
        sn2_parameters = load_family_parameters('qa_sn2_parameters.yml')
        stretch = sn2_parameters['family_parameters']['qa_sn2']['stretch']
        zmat, _ = get_stretched_sn2_zmat(xyz, {'a': 3, 'b': 1, 'e': 4, 'd': None}, sn2_parameters, mol, ring_ca_n=True)
        stretched_xyz = zmat_to_xyz(zmat)
        self.assertAlmostEqual(calculate_distance(stretched_xyz, [3, 1]), stretch * calculate_distance(xyz, [3, 1]),
                               delta=0.01)
        for bond in mol.get_all_edges():
            i, j = mol.atoms.index(bond.atom1), mol.atoms.index(bond.atom2)
            if {i, j} != {1, 3}:
                self.assertAlmostEqual(calculate_distance(stretched_xyz, [i, j]), calculate_distance(xyz, [i, j]),
                                       delta=0.05, msg=f'bond {i}-{j}')
        # Ca is planar, as in an SN2 TS
        for atom in mol.atoms[3].edges.keys():
            x = mol.atoms.index(atom)
            if x != 1:
                self.assertGreater(calculate_angle(stretched_xyz, [1, 3, x]), SN2_TS_N_CA_X_ANGLE_RANGE[0] - 2)
                self.assertLess(calculate_angle(stretched_xyz, [1, 3, x]), SN2_TS_N_CA_X_ANGLE_RANGE[1] + 2)

    def test_get_stretched_sn2_zmat_three_membered_ring(self):
        """
        Test that the planar-Ca restraint is not applied to a three-membered ring neighbour of Ca,
        which is also bonded to N: C[N+]1(C)CC1.
        """
        qa = ARCSpecies(label='QA', smiles='C[N+]1(C)CC1')
        xyz, mol = qa.get_xyz(), qa.mol
        n = [i for i, atom in enumerate(mol.atoms) if atom.is_nitrogen()][0]
        ca, cb = [mol.atoms.index(atom) for atom in mol.atoms[n].edges.keys()
                  if atom.is_carbon() and mol.is_bond_in_cycle(mol.get_bond(mol.atoms[n], atom))]
        e = [mol.atoms.index(atom) for atom in mol.atoms[ca].edges.keys() if mol.atoms.index(atom) != n][0]
        sn2_parameters = load_family_parameters('qa_sn2_parameters.yml')
        zmat, _ = get_stretched_sn2_zmat(xyz, {'a': ca, 'b': n, 'e': e, 'd': None}, sn2_parameters, mol, ring_ca_n=True)
        stretched_xyz = zmat_to_xyz(zmat)
        self.assertAlmostEqual(calculate_distance(stretched_xyz, [ca, cb]), calculate_distance(xyz, [ca, cb]),
                               delta=0.05)

    def test_get_ca_cb_staggered_rotamers_ring_cb(self):
        """
        Test the ring flip that places the hydrogen of a ring Cb, rather than its methyl group, anti to N:
        C[N+]1(C)CC(C)CCC1, Ca = 3, N = 1, Cb = 4 (ring CH), methyl C = 5 (equatorial, anti to N, in the input).
        """
        qa = ARCSpecies(label='QA', smiles='C[N+]1(C)CC(C)CCC1', xyz="""C      -2.21426964    1.05184728    0.72971984
N      -1.12555988    0.18321876    0.08484195
C      -1.66400846   -0.28619793   -1.26637075
C       0.13182887    1.04614366   -0.08336859
C       1.33541587    0.22227211   -0.55164313
C       2.57669281    1.11100079   -0.64161905
C       1.59290022   -0.95911632    0.38663053
C       0.34762126   -1.81952767    0.55444655
C      -0.85026651   -1.00104154    1.01674253
H      -1.85165662    1.40152738    1.70095182
H      -3.11335221    0.44123025    0.85586498
H      -2.41463855    1.90078678    0.06922913
H      -1.87324212    0.59615108   -1.87855906
H      -0.92357450   -0.91722618   -1.76010272
H      -2.58254394   -0.85454698   -1.09179562
H       0.33610351    1.50446147    0.89255630
H      -0.11010539    1.84188298   -0.79786826
H       1.14249241   -0.16535536   -1.55907938
H       2.41005079    1.94023113   -1.33729289
H       2.83612641    1.53478584    0.33455018
H       3.43823661    0.53916650   -1.00226414
H       2.41161380   -1.57650561   -0.00186150
H       1.91298389   -0.59289136    1.37067792
H       0.54978135   -2.60562498    1.29145962
H       0.12196481   -2.33419072   -0.38615489
H      -0.66268372   -0.58860364    2.01555199
H      -1.75791107   -1.61387772    1.05475665""")
        xyz, mol = qa.get_xyz(), qa.mol
        self.assertGreater(abs(get_angle_in_180_range(calculate_dihedral_angle(xyz, [1, 3, 4, 5], index=0))), 170)
        self.assertEqual(get_ca_cb_staggered_rotamers(xyz=xyz, mol=mol, ca=3, n=1), [])
        conformers = get_ca_cb_staggered_rotamers(xyz=xyz, mol=mol, ca=3, n=1, include_ring_cb=True)
        self.assertEqual(len(conformers), 1)
        self.assertAlmostEqual(abs(get_angle_in_180_range(calculate_dihedral_angle(conformers[0], [1, 3, 4, 17],
                                                                                   index=0))), 180, delta=5)
        self.assertLess(abs(get_angle_in_180_range(calculate_dihedral_angle(conformers[0], [1, 3, 4, 5], index=0))), 100)
        for bond in mol.get_all_edges():
            i, j = mol.atoms.index(bond.atom1), mol.atoms.index(bond.atom2)
            self.assertAlmostEqual(calculate_distance(conformers[0], [i, j]), calculate_distance(xyz, [i, j]), delta=0.05)

    def test_heuristics_for_qa_ring_opening(self):
        """
        Test heuristics for a QA SN2 ring opening: C[N+]1(C)CCCC1 + [OH-].O <=> CN(C)CCCCO.O.
        """
        qa = ARCSpecies(label='QA', smiles='C[N+]1(C)CCCC1', xyz=self.qa_pyrrolidinium_xyz)
        rxn = ARCReaction(r_species=[qa, self.oh_d1],
                          p_species=[ARCSpecies(label='amino_alcohol_water', smiles='CN(C)CCCCO.O')])
        self.assertEqual(rxn.family, 'qa_ring_opening')
        adapter = HeuristicsAdapter(job_type='tsg',
                                    reactions=[rxn],
                                    testing=True,
                                    project='test',
                                    project_directory=os.path.join(self.scratch_dir, 'heuristics_qa_ring_opening'))
        adapter.execute_incore()
        self.assertTrue(rxn.ts_species.is_ts)
        self.assertGreater(len(rxn.ts_species.ts_guesses), 0)
        reactant_xyz = qa.get_xyz()
        for tsg in rxn.ts_species.ts_guesses:
            xyz = tsg.initial_xyz
            self.assertEqual(len(xyz['symbols']), 26)
            # Ca is one of the two ring carbons bonded to N, the one O_r (atom 21) attacks
            ca = min([3, 6], key=lambda i: calculate_distance(xyz, [i, 21]))
            self.assertAlmostEqual(calculate_distance(xyz, [ca, 1]), 1.3 * calculate_distance(reactant_xyz, [ca, 1]), 2)
            for i, j in [(3, 4), (4, 5), (5, 6), (1, 0), (1, 2)]:  # ring C-C bonds and N-CH3 bonds stay intact
                self.assertLess(calculate_distance(xyz, [i, j]), 1.7)

    def test_heuristics_for_qa_sn2_with_a_triple_bond(self):
        """
        Test heuristics for QA SN2 with a C#C bond in the substrate:
        C[N+](C)(C)CCC#C + [OH-].O <=> OCCC#C.O + CN(C)C.
        The C#C bond must not be considered a collision.
        """
        rxn = ARCReaction(r_species=[self.butynyl_tma, self.oh_d1],
                          p_species=[self.butynol_water, self.trimethylamine],
                          family='qa_sn2')
        adapter = HeuristicsAdapter(job_type='tsg',
                                    reactions=[rxn],
                                    testing=True,
                                    project='test',
                                    project_directory=os.path.join(self.scratch_dir, 'heuristics_qasn2_triple_bond'))
        with mock.patch.dict(settings, {'UMA_TS_GUESS_PREFILTER': False}):
            adapter.execute_incore()
        self.assertEqual(rxn.family, 'qa_sn2')
        self.assertTrue(rxn.ts_species.is_ts)
        self.assertGreater(len(rxn.ts_species.ts_guesses), 0)
        symbols = rxn.ts_species.ts_guesses[0].initial_xyz['symbols']
        self.assertEqual(len(symbols), 27)
        self.assertEqual(sorted(symbols[22:]), ['H', 'H', 'H', 'O', 'O'])

    def test_get_ca_cb_staggered_rotamers(self):
        """Test the get_ca_cb_staggered_rotamers() function and its use as the qa_sn2 Ca-Cb rotamer fallback."""
        n_idx, ca_idx, cb_idx = 1, 4, 5
        # A quaternary Cb bears three substituents; none is anti to N in the input conformer by less than
        # the tolerance, so each one placed anti to N gives a distinct staggered rotamer.
        xyz = self.neopentyl_tma.get_xyz()
        mol = self.neopentyl_tma.mol
        cb_side = [5, 6, 7, 8] + list(range(20, 29))
        ca_side = [i for i in range(len(xyz['symbols'])) if i not in cb_side]
        cb_side_bonds = [(mol.atoms.index(bond.atom1), mol.atoms.index(bond.atom2)) for bond in mol.get_all_edges()
                         if {mol.atoms.index(bond.atom1), mol.atoms.index(bond.atom2)} <= set(cb_side + [ca_idx])]

        def min_nonbonded_h_h(coords: tuple) -> float:
            """The shortest distance between H atoms bonded to different atoms."""
            h_parent = {i: mol.atoms.index(list(atom.edges.keys())[0])
                        for i, atom in enumerate(mol.atoms) if atom.is_hydrogen()}
            return min(calculate_distance(coords, [i, j], index=0)
                       for i, j in itertools.combinations(h_parent.keys(), 2) if h_parent[i] != h_parent[j])

        # If the force field cannot be set up, the rigid rotamers are kept: the Ca side does not move,
        # and the Cb side is rotated rigidly.
        with mock.patch('arc.job.adapters.ts.heuristics_qa.AllChem.MMFFGetMoleculeProperties', return_value=None):
            rigid_rotamers = get_ca_cb_staggered_rotamers(xyz=xyz, mol=mol, ca=ca_idx, n=n_idx)
        self.assertEqual(len(rigid_rotamers), 3)
        for rotamer, x_idx in zip(rigid_rotamers, [6, 7, 8]):
            self.assertAlmostEqual(calculate_dihedral_angle(rotamer, [n_idx, ca_idx, cb_idx, x_idx], index=0),
                                   180.0, places=3)
            for i in ca_side:
                self.assertTrue(np.allclose(rotamer['coords'][i], xyz['coords'][i]))
            for i, j in itertools.combinations(cb_side + [ca_idx], 2):
                self.assertAlmostEqual(calculate_distance(rotamer['coords'], [i, j], index=0),
                                       calculate_distance(xyz['coords'], [i, j], index=0), places=4)

        # By default each rotamer is relaxed with the chosen dihedral constrained at 180°:
        # the Cb-side bond lengths are preserved, and close nonbonded H...H contacts are relieved.
        rotamers = get_ca_cb_staggered_rotamers(xyz=xyz, mol=mol, ca=ca_idx, n=n_idx)
        self.assertEqual(len(rotamers), 3)
        self.assertEqual(rotamers, get_ca_cb_staggered_rotamers(xyz=xyz, mol=mol, ca=ca_idx, n=n_idx))
        for rotamer, rigid_rotamer, x_idx in zip(rotamers, rigid_rotamers, [6, 7, 8]):
            self.assertEqual(rotamer['symbols'], xyz['symbols'])
            self.assertEqual(rotamer['isotopes'], xyz['isotopes'])
            self.assertAlmostEqual(calculate_dihedral_angle(rotamer, [n_idx, ca_idx, cb_idx, x_idx], index=0),
                                   180.0, places=1)
            for i, j in cb_side_bonds:
                self.assertAlmostEqual(calculate_distance(rotamer['coords'], [i, j], index=0),
                                       calculate_distance(xyz['coords'], [i, j], index=0), delta=0.05)
            self.assertGreaterEqual(min_nonbonded_h_h(rotamer['coords']), min_nonbonded_h_h(rigid_rotamer['coords']))

        # A Ca-Cb bond in a ring is not rotatable: no rotamers.
        pyrrolidinium = ARCSpecies(label='dimethylpyrrolidinium', smiles='C[N+]1(C)CCCC1')
        n_idx = [i for i, atom in enumerate(pyrrolidinium.mol.atoms) if atom.is_nitrogen()][0]
        ring_ca_idx = [pyrrolidinium.mol.atoms.index(atom) for atom in pyrrolidinium.mol.atoms[n_idx].edges.keys()
                       if pyrrolidinium.mol.is_bond_in_cycle(pyrrolidinium.mol.get_bond(atom,
                                                                                       pyrrolidinium.mol.atoms[n_idx]))][0]
        self.assertEqual(get_ca_cb_staggered_rotamers(xyz=pyrrolidinium.get_xyz(), mol=pyrrolidinium.mol,
                                                      ca=ring_ca_idx, n=n_idx), [])

        # Ca without a heavy-atom neighbor other than N: no rotamers.
        self.assertEqual(get_ca_cb_staggered_rotamers(xyz=self.tma.get_xyz(), mol=self.tma.mol, ca=0, n=1), [])

        # sp2 Cb: the carbonyl O is not anti to N, but rotating it there swings the NMe2 group into
        # the NMe3 group, so the only candidate rotamer collides and is discarded.
        self.assertEqual(get_ca_cb_staggered_rotamers(xyz=self.amide_alpha_tma.get_xyz(), mol=self.amide_alpha_tma.mol,
                                                      ca=4, n=1), [])

        # The fallback is not attempted when the input conformer already yields guesses.
        rxn = ARCReaction(r_species=[self.tma, self.oh_d1],
                          p_species=[self.methanol_water, self.trimethylamine],
                          family='qa_sn2')
        with mock.patch('arc.job.adapters.ts.heuristics_qa.get_ca_cb_staggered_rotamers') as rotamers_mock, \
                mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                           side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
            xyz_guesses, _ = qa_sn2(rxn)
        self.assertGreater(len(xyz_guesses), 0)
        rotamers_mock.assert_not_called()

        # Beta-branched substrates whose input conformer has a Cb substituent in the backside-attack path:
        # qa_sn2 (without the UMA pre-filter) attempts the fallback, and every guess produced must be valid.
        def run_beta_branched_case(substrate: ARCSpecies, alcohol_water: ARCSpecies) -> list[dict]:
            rxn = ARCReaction(r_species=[substrate, self.oh_d1],
                              p_species=[alcohol_water, self.trimethylamine],
                              family='qa_sn2')
            with mock.patch('arc.job.adapters.ts.heuristics_qa.get_ca_cb_staggered_rotamers',
                            wraps=get_ca_cb_staggered_rotamers) as rotamers_mock, \
                    mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                               side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
                xyz_guesses, _ = qa_sn2(rxn)
            rotamers_mock.assert_called()
            n_qa = len(substrate.mol.atoms)
            for xyz_guess in xyz_guesses:
                xyz_guess = self.to_construction_order(xyz_guess, n_qa)
                self.assertEqual(len(xyz_guess['symbols']), n_qa + 5)
                self.assertEqual(xyz_guess['symbols'][:n_qa], substrate.get_xyz()['symbols'])
                self.assertTrue(check_sn2_ts_bonds(xyz_guess, [n_qa, n_qa + 1, 4, n_qa + 2, n_qa + 3]))
            return xyz_guesses

        # A ring carbon of Cb blocks the backside of Ca: C[N+](C)(C)CC1CCCC1 + [OH-].O <=> OCC1CCCC1.O + CN(C)C.
        self.assertGreater(len(run_beta_branched_case(self.cyclopentylmethyl_tma, self.cyclopentylmethanol_water)), 0)
        # A ring carbon of Cb blocks the backside of Ca: C[N+](C)(C)CC1CCCCC1 + [OH-].O <=> OCC1CCCCC1.O + CN(C)C.
        self.assertGreater(len(run_beta_branched_case(self.cyclohexylmethyl_tma, self.cyclohexylmethanol_water)), 0)
        # Neopentyl (quaternary Cb): C[N+](C)(C)CC(C)(C)C + [OH-].O <=> OCC(C)(C)C.O + CN(C)C.
        # Every staggered Ca-Cb rotamer has a methyl anti to N, i.e., in the backside-attack path,
        # so guesses are not required here.
        run_beta_branched_case(self.neopentyl_tma, self.neopentanol_water)

    def test_qa_ts_guesses_follow_reactant_order(self):
        """
        Test that QA TS guesses list their atoms in the concatenated reactant order of the reaction
        (the QA atoms, then the OH_d1 atoms in their own order), so the TS checks that use reactant
        atom indices (e.g., the normal mode displacement check) can run.
        """
        cases = [(qa_sn2, self.tma, ARCReaction(r_species=[self.tma, self.oh_d1],
                                                p_species=[self.methanol_water, self.trimethylamine],
                                                family='qa_sn2')),
                 (qa_hofmann, self.ethyl_tma, ARCReaction(r_species=[self.ethyl_tma, self.oh_d1],
                                                          p_species=[self.ethene, self.trimethylamine,
                                                                     self.water_dimer])),
                 ]
        oh_d1_bonds = [(self.oh_d1.mol.atoms.index(atom_1), self.oh_d1.mol.atoms.index(atom_2))
                       for atom_1 in self.oh_d1.mol.atoms for atom_2 in atom_1.edges
                       if self.oh_d1.mol.atoms.index(atom_1) < self.oh_d1.mol.atoms.index(atom_2)]
        for tsg_func, qa, rxn in cases:
            with self.subTest(family=tsg_func.__name__):
                with mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                                side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
                    xyz_guesses, _ = tsg_func(rxn)
                self.assertGreater(len(xyz_guesses), 0)
                reactant_symbols = rxn.get_reactants_xyz(return_format='dict')['symbols']
                n_qa = len(qa.mol.atoms)
                for xyz_guess in xyz_guesses:
                    self.assertEqual(tuple(xyz_guess['symbols']), tuple(reactant_symbols))
                    # Each covalent O-H bond of OH_d1 is found at its OH_d1 indices, offset by the QA atoms.
                    for i, j in oh_d1_bonds:
                        self.assertLess(calculate_distance(xyz_guess['coords'], [n_qa + i, n_qa + j], index=0), 1.2)

    def test_qa_ts_guesses_with_oh_d1_listed_first(self):
        """
        Test that a QA reaction with the OH_d1 species listed first gets the same TS guesses as with the QA reactant
        listed first, with the atoms listed in the user's reactant order.
        """
        cases = [(qa_sn2, self.tma, [self.methanol_water, self.trimethylamine]),
                 (qa_hofmann, self.ethyl_tma, [self.ethene, self.trimethylamine, self.water_dimer]),
                 ]
        n_oh_d1 = len(self.oh_d1.mol.atoms)
        for tsg_func, qa, products in cases:
            with self.subTest(family=tsg_func.__name__):
                with mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                                side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
                    qa_first_guesses, _ = tsg_func(ARCReaction(r_species=[qa, self.oh_d1], p_species=products))
                    rxn = ARCReaction(r_species=[self.oh_d1, qa], p_species=products)
                    oh_d1_first_guesses, _ = tsg_func(rxn)
                self.assertGreater(len(oh_d1_first_guesses), 0)
                self.assertEqual(len(oh_d1_first_guesses), len(qa_first_guesses))
                reactants_xyz = rxn.get_reactants_xyz(return_format='dict')
                n_qa = len(qa.mol.atoms)
                for qa_first, oh_d1_first in zip(qa_first_guesses, oh_d1_first_guesses):
                    self.assertEqual(tuple(oh_d1_first['symbols']), tuple(reactants_xyz['symbols']))
                    self.assertEqual(tuple(oh_d1_first['isotopes']), tuple(reactants_xyz['isotopes']))
                    self.assertEqual(tuple(oh_d1_first['coords'][:n_oh_d1]), tuple(qa_first['coords'][n_qa:]))
                    self.assertEqual(tuple(oh_d1_first['coords'][n_oh_d1:]), tuple(qa_first['coords'][:n_qa]))

    def test_qa_ts_guesses_for_a_reaction_written_in_reverse(self):
        """Test that a QA reaction written in reverse (products first) gets a clear error rather than guesses."""
        rxn = ARCReaction(r_species=[self.methanol_water, self.trimethylamine], p_species=[self.tma, self.oh_d1])
        self.assertEqual(rxn.family, 'qa_sn2')
        with self.assertRaisesRegex(ValueError, 'only in reverse'):
            qa_sn2(rxn)

    def test_qa_ts_guesses_keep_reactant_isotopes(self):
        """
        Test that QA TS guesses keep the isotope labels of both reactants (a deuterium on the QA and on the water).
        """
        def deuterate(spc: ARCSpecies, index: int, label: str) -> ARCSpecies:
            xyz = dict(spc.get_xyz())
            isotopes = list(xyz['isotopes'])
            isotopes[index] = 2
            xyz['isotopes'] = tuple(isotopes)
            return ARCSpecies(label=label, smiles=spc.mol.copy(deep=True).to_smiles(), xyz=xyz)

        qa_h = next(i for i, symbol in enumerate(self.tma.get_xyz()['symbols']) if symbol == 'H')
        water_h = next(i for i, atom in enumerate(self.oh_d1.mol.atoms)
                       if atom.is_hydrogen() and list(atom.edges)[0].charge == 0)
        rxn = ARCReaction(r_species=[deuterate(self.tma, qa_h, 'TMA_d'), deuterate(self.oh_d1, water_h, 'OH_d1_d')],
                          p_species=[self.methanol_water, self.trimethylamine],
                          family='qa_sn2')
        reactant_isotopes = rxn.get_reactants_xyz(return_format='dict')['isotopes']
        self.assertEqual(reactant_isotopes.count(2), 2)
        with mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                        side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
            xyz_guesses, _ = qa_sn2(rxn)
        self.assertGreater(len(xyz_guesses), 0)
        for xyz_guess in xyz_guesses:
            self.assertEqual(tuple(xyz_guess['isotopes']), tuple(reactant_isotopes))

    def test_load_qa_hofmann_parameters(self):
        """
        Test that load_family_parameters correctly loads the QA Hofmann elimination YAML.
        """
        hofmann = load_family_parameters("qa_hofmann_parameters.yml")
        self.assertIn('qa_hofmann', hofmann['family_parameters'])
        qa = hofmann['family_parameters']['qa_hofmann']
        self.assertEqual(len(qa['r_value']), 5)
        self.assertEqual(len(qa['a_value']), 5)
        self.assertEqual(len(qa['d_values']), 4)
        for row in qa['d_values']:
            self.assertEqual(len(row), 5)
            self.assertEqual(row[:2], ['scan', 'scan'])
        self.assertEqual(set(qa['stretch'].keys()), {'Ca-N', 'Ca-Cb', 'Cb-Hb'})
        self.assertEqual(set(qa['pi_activated']['stretch'].keys()), {'Ca-N', 'Ca-Cb', 'Cb-Hb'})
        self.assertIsInstance(qa['pi_activated']['r_Hb-Or'], float)

    def test_check_hofmann_ts_bonds(self):
        """
        Test check_hofmann_ts_bonds with passing and failing geometries.
        """
        # Atom layout used throughout:
        #   0=Cb, 1=Ca, 2=Hb, 3=O_r, 4=H_r, 5=H1_w, 6=O_w, 7=H2_w
        # atom_indices = [O_r_seq=3, H_r_seq=4, Cb_xyz=0, H1_w_seq=5, O_w_seq=6, Hb_xyz=2]
        INDICES = [3, 4, 0, 5, 6, 2]

        def make_xyz(coords):
            return {'symbols': ('C', 'C', 'H', 'O', 'H', 'H', 'O', 'H'),
                    'isotopes': (12, 12, 1, 16, 1, 1, 16, 1),
                    'coords': coords}

        valid_coords = (
            ( 0.0,   0.0,  0.0),
            ( 1.46,  0.0,  0.0),
            (-1.53,  0.0,  0.0),
            (-2.66,  0.0,  0.0),
            (-2.90,  0.93, 0.0),
            (-3.10, -1.67, 0.0),
            (-3.35, -2.63, 0.0),
            (-4.30, -2.70, 0.0),
        )
        self.assertTrue(check_hofmann_ts_bonds(make_xyz(valid_coords), INDICES))

        # Failing: Hb fully transferred to O_r (Hb's two nearest atoms are O_r and H_r, not Cb)
        fail_hb = list(valid_coords)
        fail_hb[2] = (-2.2, 0.4, 0.0)
        self.assertFalse(check_hofmann_ts_bonds(make_xyz(tuple(fail_hb)), INDICES))

        # Failing: H_r not bonded to O_r (the QA SN2 water/hydroxide checks also apply)
        fail_hr = list(valid_coords)
        fail_hr[4] = (-1.53, 1.0, 0.0)
        self.assertFalse(check_hofmann_ts_bonds(make_xyz(tuple(fail_hr)), INDICES))

    def test_stretch_xyz_bond(self):
        """Test stretching a bond in Cartesian coordinates by translating one side of the bond."""
        xyz = self.ethyl_tma.get_xyz()
        mol = self.ethyl_tma.mol
        ca, n, cb, hb = 1, 2, 0, 6
        stretched = stretch_xyz_bond(xyz, mol, ca, n, 1.05)
        self.assertAlmostEqual(calculate_distance(stretched['coords'], [ca, n], index=0),
                               1.05 * calculate_distance(xyz['coords'], [ca, n], index=0), places=5)
        # The N side (N, its methyl carbons and hydrogens) is translated rigidly, the Ca side does not move.
        for i, j in itertools.combinations([2, 3, 4, 5] + list(range(11, 20)), 2):
            self.assertAlmostEqual(calculate_distance(stretched['coords'], [i, j], index=0),
                                   calculate_distance(xyz['coords'], [i, j], index=0), places=5)
        for i in [0, 1, 6, 7, 8, 9, 10]:
            self.assertTrue(np.allclose(stretched['coords'][i], xyz['coords'][i]))
        # A terminal hydrogen: only Hb moves.
        stretched = stretch_xyz_bond(xyz, mol, cb, hb, 1.405)
        self.assertAlmostEqual(calculate_distance(stretched['coords'], [cb, hb], index=0),
                               1.405 * calculate_distance(xyz['coords'], [cb, hb], index=0), places=5)
        for i in range(len(xyz['symbols'])):
            if i != hb:
                self.assertTrue(np.allclose(stretched['coords'][i], xyz['coords'][i]))
        # A ring bond: only the translated atom and its hydrogens move.
        pyrrolidinium = ARCSpecies(label='dimethylpyrrolidinium', smiles='C[N+]1(C)CCCC1')
        p_mol, p_xyz = pyrrolidinium.mol, pyrrolidinium.get_xyz()
        n_idx = [i for i, atom in enumerate(p_mol.atoms) if atom.is_nitrogen()][0]
        ring_c = [p_mol.atoms.index(atom) for atom in p_mol.atoms[n_idx].edges.keys()
                  if p_mol.is_bond_in_cycle(p_mol.get_bond(atom, p_mol.atoms[n_idx]))][0]
        moved = [n_idx]
        stretched = stretch_xyz_bond(p_xyz, p_mol, ring_c, n_idx, 1.05)
        for i in range(len(p_xyz['symbols'])):
            self.assertEqual(np.allclose(stretched['coords'][i], p_xyz['coords'][i]), i not in moved)

    def test_is_pi_activated_beta_carbon(self):
        """Test detecting a beta carbon conjugated to a pi system."""
        def is_activated(spc: ARCSpecies, cb: int, ca: int) -> bool:
            return is_pi_activated_beta_carbon(spc.mol, cb=cb, ca=ca, multiple_bond_pairs=get_multiple_bond_pairs(spc))

        # Aryl (Kekule ring bond on the ipso carbon), C=O, C=C, and C#N neighbors of Cb.
        self.assertTrue(is_activated(self.phenethyl_tma, cb=5, ca=4))
        self.assertTrue(is_activated(ARCSpecies(label='oxobutyl_TMA', smiles='CC(=O)CC[N+](C)(C)C'), cb=3, ca=4))
        self.assertTrue(is_activated(ARCSpecies(label='butenyl_TMA', smiles='C=CCC[N+](C)(C)C'), cb=2, ca=3))
        self.assertTrue(is_activated(ARCSpecies(label='cyanoethyl_TMA', smiles='N#CCC[N+](C)(C)C'), cb=2, ca=3))
        # Saturated beta carbons.
        self.assertFalse(is_activated(self.ethyl_tma, cb=0, ca=1))
        self.assertFalse(is_activated(self.propyl_tma, cb=1, ca=2))
        self.assertFalse(is_activated(ARCSpecies(label='isopropyl_TMA', smiles='CC(C)[N+](C)(C)C'), cb=0, ca=1))

    def test_get_unique_hofmann_product_dicts(self):
        """Test removing product dictionaries that abstract an equivalent beta hydrogen."""
        def get_product_dicts(qa: ARCSpecies, alkene: ARCSpecies, amine: ARCSpecies) -> list[dict]:
            rxn = ARCReaction(r_species=[qa, self.oh_d1], p_species=[alkene, amine, self.water_dimer])
            return [pd for pd in get_reaction_family_products(rxn=rxn, rmg_family_set="default",
                                                              consider_rmg_families=False, consider_arc_families=True)
                    if pd.get('family') == 'qa_hofmann']

        # The three hydrogens of a methyl Cb are interchanged by the methyl rotation.
        product_dicts = get_product_dicts(self.ethyl_tma, self.ethene, self.trimethylamine)
        self.assertEqual(len(product_dicts), 3)
        self.assertEqual(len(get_unique_hofmann_product_dicts(product_dicts, self.ethyl_tma)), 1)
        # Equivalent ethyl groups: a single (Ca, Cb) pair and a single Hb are kept.
        tetraethylammonium = ARCSpecies(label='Et4N', smiles='CC[N+](CC)(CC)CC')
        product_dicts = get_product_dicts(tetraethylammonium, self.ethene, ARCSpecies(label='Et3N', smiles='CCN(CC)CC'))
        self.assertEqual(len(product_dicts), 12)
        self.assertEqual(len(get_unique_hofmann_product_dicts(product_dicts, tetraethylammonium)), 1)
        # Both hydrogens of a methylene Cb are kept.
        product_dicts = get_product_dicts(self.propyl_tma, self.propene, self.trimethylamine)
        unique_product_dicts = get_unique_hofmann_product_dicts(product_dicts, self.propyl_tma)
        self.assertEqual(sorted(pd['r_label_map']['*4'] for pd in unique_product_dicts), [10, 11])
        # Achiral isopropyl group: its two methyl Cb are equivalent, a single (Ca, Cb) pair is kept.
        isopropyl_tma = ARCSpecies(label='isopropyl_TMA', smiles='CC(C)[N+](C)(C)C')
        product_dicts = get_product_dicts(isopropyl_tma, self.propene, self.trimethylamine)
        self.assertEqual(len(product_dicts), 6)
        unique_product_dicts = get_unique_hofmann_product_dicts(product_dicts, isopropyl_tma)
        self.assertEqual(len(unique_product_dicts), 1)
        # Chiral QA reactant: the two isopropyl methyl Cb are diastereotopic, both (Ca, Cb) pairs are kept.
        chiral_qa = ARCSpecies(label='chiral_QA', smiles='C[N+](C)(C(C)C)[C@@H](C)c1ccccc1')
        product_dicts = get_product_dicts(chiral_qa, self.propene,
                                          ARCSpecies(label='amine', smiles='CN(C)[C@@H](C)c1ccccc1'))
        self.assertEqual(len(product_dicts), 6)
        unique_product_dicts = get_unique_hofmann_product_dicts(product_dicts, chiral_qa)
        self.assertEqual(sorted(pd['r_label_map']['*3'] for pd in unique_product_dicts), [4, 5])

    def test_get_hofmann_anti_periplanar_conformers(self):
        """Test getting QA conformers with the beta hydrogen anti-periplanar to N."""
        # Both beta hydrogens of the propyl conformer are gauche to N: the Ca-Cb bond is rotated.
        xyz, mol = self.propyl_tma.get_xyz(), self.propyl_tma.mol
        for hb in [10, 11]:
            xyz_indices = {'Ca': 2, 'N': 3, 'Cb': 1, 'Hb': hb}
            self.assertLess(abs(calculate_dihedral_angle(xyz, [3, 2, 1, hb], index=0) - 180.0), 150.0)
            conformers = get_hofmann_anti_periplanar_conformers(xyz, mol, xyz_indices)
            self.assertGreater(len(conformers), 0)
            for conformer in conformers:
                self.assertAlmostEqual(calculate_dihedral_angle(conformer, [3, 2, 1, hb], index=0), 180.0, delta=1.0)
        # A beta hydrogen already anti to N: the input conformer is kept.
        xyz, mol = self.ethyl_tma.get_xyz(), self.ethyl_tma.mol
        anti_hb = [hb for hb in [6, 7, 8] if abs(calculate_dihedral_angle(xyz, [2, 1, 0, hb], index=0) - 180.0) < 30][0]
        conformers = get_hofmann_anti_periplanar_conformers(xyz, mol, {'Ca': 1, 'N': 2, 'Cb': 0, 'Hb': anti_hb})
        self.assertEqual(conformers, [xyz])

        # Bulky groups on both Ca and Cb (3,3-dimethylbutyl-NMe3+), both beta hydrogens gauche to N:
        # every rigid Ca-Cb rotamer collides (tert-butyl and NMe3 hydrogens), but the relaxed rotamers do not.
        xyz = """C       2.86192523    0.62166789    1.24961510
C       1.98798079    0.07220836    0.10447799
C       2.31323189    0.85876414   -1.17621980
C       2.33947467   -1.40969458   -0.10779435
C       0.49571102    0.24513249    0.50827851
C      -0.51655683   -0.25547806   -0.52974249
N      -1.98825426   -0.06935983   -0.09842311
C      -2.87278287   -0.61139613   -1.22754007
C      -2.30605903   -0.85302234    1.16977539
C      -2.33214199    1.40120490    0.10805981
H       2.67513625    0.08113304    2.18442266
H       3.92869001    0.52483722    1.01781765
H       2.65660207    1.68291004    1.43000417
H       3.38648265    0.82538023   -1.39666838
H       1.79301539    0.44920329   -2.04812057
H       2.02747489    1.91175059   -1.07623248
H       1.81938566   -1.83090031   -0.97420593
H       2.07283473   -2.00847024    0.77016071
H       3.41382195   -1.53805388   -0.28351879
H       0.33872084   -0.28109617    1.45697609
H       0.32033116    1.30854024    0.70826642
H      -0.40654947    0.28371076   -1.47648064
H      -0.38791804   -1.32651453   -0.71808122
H      -2.63941723   -1.67066557   -1.37063733
H      -3.92102541   -0.48773537   -0.93922567
H      -2.65825698   -0.04283589   -2.13732378
H      -1.99037635   -1.89002444    1.02440229
H      -1.77745047   -0.39893540    2.01138006
H      -3.38507304   -0.80258278    1.34544079
H      -1.80255573    1.77086381    0.98942898
H      -2.03487038    1.95538752   -0.78674418
H      -3.41153113    1.48407099    0.26845217"""
        spc = ARCSpecies(label='dimethylbutyl_TMA', smiles='CC(C)(C)CC[N+](C)(C)C', xyz=xyz)
        xyz, mol = spc.get_xyz(), spc.mol
        self.assertEqual(get_ca_cb_staggered_rotamers(xyz=xyz, mol=mol, ca=5, n=6), [])
        for hb in [19, 20]:
            self.assertLess(abs(get_angle_in_180_range(calculate_dihedral_angle(xyz, [6, 5, 4, hb], index=0))), 90.0)
            conformers = get_hofmann_anti_periplanar_conformers(xyz, mol, {'Ca': 5, 'N': 6, 'Cb': 4, 'Hb': hb})
            self.assertGreater(len(conformers), 0)
            for conformer in conformers:
                self.assertAlmostEqual(abs(get_angle_in_180_range(
                    calculate_dihedral_angle(conformer, [6, 5, 4, hb], index=0))), 180.0, delta=1.0)
                self.assertFalse(colliding_atoms(conformer))

    def test_get_hofmann_anti_periplanar_conformers_in_a_ring(self):
        """
        Test that for a ring Ca-Cb bond, which cannot be rotated, a ring conformer with Hb anti to N is used
        (e.g., the cyclohexyl chair with N axial), keeping the stereochemistry of the input conformer.
        If no ring conformer has Hb anti to N (e.g., cyclopentyl), the best ring conformer is used with a warning.
        """
        def dihedral(xyz: dict, indices: list[int]) -> float:
            return abs(get_angle_in_180_range(calculate_dihedral_angle(xyz, indices, index=0)))

        def stereo(spc: ARCSpecies, xyz: dict) -> str:
            _, rd_mol = rdkit_conf_from_mol(spc.mol, xyz)
            Chem.AssignStereochemistryFrom3D(rd_mol)
            return get_rdkit_stereo_smiles(rd_mol)

        # Cyclohexyl-NMe3+ (N 1, Ca 4, ring Cb 5): the input conformer has N equatorial, so no Hb on Cb is anti to N.
        cyclohexyl_tma = ARCSpecies(label='cyclohexyl_TMA', smiles='C[N+](C)(C)C1CCCCC1')
        xyz, mol = cyclohexyl_tma.get_xyz(), cyclohexyl_tma.mol
        hbs = [mol.atoms.index(atom) for atom in mol.atoms[5].edges.keys() if atom.is_hydrogen()]
        self.assertEqual(len(hbs), 2)
        for hb in hbs:
            self.assertLess(dihedral(xyz, [1, 4, 5, hb]), 150.0)
        conformers = {hb: get_hofmann_anti_periplanar_conformers(xyz, mol, {'Ca': 4, 'N': 1, 'Cb': 5, 'Hb': hb})
                      for hb in hbs}
        best_dihedrals = list()
        for hb in hbs:
            self.assertEqual(len(conformers[hb]), 1)
            self.assertGreater(dihedral(conformers[hb][0], [1, 4, 5, hb]), dihedral(xyz, [1, 4, 5, hb]))
            self.assertEqual(stereo(cyclohexyl_tma, conformers[hb][0]), stereo(cyclohexyl_tma, xyz))
            best_dihedrals.append(dihedral(conformers[hb][0], [1, 4, 5, hb]))
        self.assertGreaterEqual(max(best_dihedrals), 150.0)  # the axial Hb of the N-axial chair

        # All TS guesses of the axial Hb are built from the N-axial chair
        anti_hb = hbs[int(np.argmax(best_dihedrals))]
        rxn = ARCReaction(r_species=[cyclohexyl_tma, self.oh_d1],
                          p_species=[ARCSpecies(label='cyclohexene', smiles='C1=CCCCC1'),
                                     self.trimethylamine, self.water_dimer])
        self.assertEqual(rxn.family, 'qa_hofmann')
        with mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                        side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
            xyz_guesses, _ = qa_hofmann(rxn)
        n_qa = len(mol.atoms)
        o_r = n_qa + next(i for i, atom in enumerate(self.oh_d1.mol.atoms) if atom.charge < 0)
        anti_hb_guesses = [guess for guess in xyz_guesses
                           if min((i for i, symbol in enumerate(guess['symbols'][:n_qa]) if symbol == 'H'),
                                  key=lambda i: calculate_distance(guess['coords'], [i, o_r], index=0)) == anti_hb]
        self.assertGreater(len(anti_hb_guesses), 0)
        for guess in anti_hb_guesses:
            self.assertGreaterEqual(dihedral(guess, [1, 4, 5, anti_hb]), 150.0)

        # Cyclopentyl-NMe3+ (N 1, Ca 4, ring Cb 5): the envelope conformers bring Hb closer to anti, but not within 30 degrees
        cyclopentyl_tma = ARCSpecies(label='cyclopentyl_TMA', smiles='C[N+](C)(C)C1CCCC1')
        xyz, mol = cyclopentyl_tma.get_xyz(), cyclopentyl_tma.mol
        hbs = [mol.atoms.index(atom) for atom in mol.atoms[5].edges.keys() if atom.is_hydrogen()]
        improved = list()
        for hb in hbs:
            with self.assertLogs('arc', level='WARNING') as logs:
                conformers = get_hofmann_anti_periplanar_conformers(xyz, mol, {'Ca': 4, 'N': 1, 'Cb': 5, 'Hb': hb})
            self.assertTrue(any('No anti-periplanar beta hydrogen' in message for message in logs.output))
            self.assertEqual(len(conformers), 1)
            self.assertLess(dihedral(conformers[0], [1, 4, 5, hb]), 150.0)
            improved.append(dihedral(conformers[0], [1, 4, 5, hb]) > dihedral(xyz, [1, 4, 5, hb]) + 30.0)
        self.assertTrue(any(improved))

    def test_get_stereo_consistent_xyz(self):
        """Test that the QA reactant coordinates are made consistent with the stereochemistry of its SMILES."""
        trans_smiles, cis_smiles = 'C[C@@H]1CCCC[C@H]1[N+](C)(C)C', 'C[C@H]1CCCC[C@H]1[N+](C)(C)C'
        trans_qa, cis_qa = ARCSpecies(label='trans', smiles=trans_smiles), ARCSpecies(label='cis', smiles=cis_smiles)

        def stereo(xyz: dict) -> str:
            _, rd_mol = rdkit_conf_from_mol(cis_qa.mol, xyz)
            Chem.AssignStereochemistryFrom3D(rd_mol)
            return get_rdkit_stereo_smiles(rd_mol)

        canonical = {smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in [trans_smiles, cis_smiles]}
        cis_xyz = cis_qa.get_xyz()
        self.assertEqual(stereo(cis_xyz), canonical[cis_smiles])
        # Consistent, or no stereo marks: the coordinates are returned unchanged
        self.assertIs(get_stereo_consistent_xyz(cis_smiles, cis_qa.mol, cis_xyz), cis_xyz)
        self.assertIs(get_stereo_consistent_xyz('CC1CCCCC1[N+](C)(C)C', cis_qa.mol, cis_xyz), cis_xyz)
        self.assertIs(get_stereo_consistent_xyz(None, cis_qa.mol, cis_xyz), cis_xyz)
        # Inconsistent: a conformer with the SMILES stereochemistry, in the same atom order
        with self.assertLogs('arc', level='WARNING') as logs:
            trans_xyz = get_stereo_consistent_xyz(trans_smiles, trans_qa.mol, cis_xyz, label='trans')
        self.assertTrue(any('do not have the stereochemistry' in message for message in logs.output))
        self.assertEqual(stereo(trans_xyz), canonical[trans_smiles])
        self.assertEqual(trans_xyz['symbols'], cis_xyz['symbols'])

        # The TS guesses of the trans QA reactant are trans, also if its default coordinates are not
        rxn = ARCReaction(r_species=[trans_qa, self.oh_d1],
                          p_species=[ARCSpecies(label='methylcyclohexene', smiles='CC1=CCCCC1'),
                                     self.trimethylamine, self.water_dimer])
        with mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                        side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
            xyz_guesses, _ = qa_hofmann(rxn)
        self.assertGreater(len(xyz_guesses), 0)
        n_qa = len(trans_qa.mol.atoms)
        for guess in xyz_guesses:
            qa_xyz = {'symbols': guess['symbols'][:n_qa], 'isotopes': guess['isotopes'][:n_qa],
                      'coords': guess['coords'][:n_qa]}
            self.assertEqual(stereo(qa_xyz), canonical[trans_smiles])

    def to_construction_order(self, xyz: dict, n_qa: int) -> dict:
        """
        Reorder a QA TS guess (QA atoms, then the OH_d1 atoms in their own order) to its construction order
        (QA atoms, then O_r, H_r, H1_w, O_w, H2_w), the order the QA TS bond checks expect.
        """
        local_o_r = next(i for i, atom in enumerate(self.oh_d1.mol.atoms) if atom.charge < 0)
        atom_order = get_oh_d1_atom_order(self.oh_d1, local_o_r)
        return sort_xyz_using_indices(xyz, list(range(n_qa)) + [n_qa + i for i in atom_order])

    def check_hofmann_guesses(self,
                              xyz_guesses: list[dict],
                              qa: ARCSpecies,
                              ca: int,
                              n: int,
                              cb: int,
                              hb_options: list[int],
                              pi_activated: bool = False):
        """Check the reaction centre of QA Hofmann elimination TS guesses."""
        n_qa = len(qa.mol.atoms)
        o_r = n_qa
        r_hb_or, r_cb_hb = (1.26, (1.30, 1.42)) if pi_activated else (1.13, (1.48, 1.58))
        self.assertGreater(len(xyz_guesses), 0)
        for xyz_guess in xyz_guesses:
            xyz_guess = self.to_construction_order(xyz_guess, n_qa)
            self.assertEqual(len(xyz_guess['symbols']), n_qa + 5)
            self.assertEqual(xyz_guess['symbols'][:n_qa], qa.get_xyz()['symbols'])
            self.assertEqual(sorted(xyz_guess['symbols'][n_qa:]), ['H', 'H', 'H', 'O', 'O'])
            hb = min(hb_options, key=lambda h: calculate_distance(xyz_guess['coords'], [h, o_r], index=0))
            self.assertTrue(check_hofmann_ts_bonds(xyz_guess, [o_r, o_r + 1, cb, o_r + 2, o_r + 3, hb]))
            self.assertAlmostEqual(calculate_distance(xyz_guess['coords'], [hb, o_r], index=0), r_hb_or, places=2)
            self.assertAlmostEqual(calculate_distance(xyz_guess['coords'], [o_r, o_r + 2], index=0), 1.72, places=2)
            self.assertTrue(1.50 < calculate_distance(xyz_guess['coords'], [ca, n], index=0) < 1.72)
            self.assertTrue(1.43 < calculate_distance(xyz_guess['coords'], [ca, cb], index=0) < 1.52)
            self.assertTrue(r_cb_hb[0] < calculate_distance(xyz_guess['coords'], [cb, hb], index=0) < r_cb_hb[1])
            # E2: Hb is anti-periplanar to the leaving N.
            self.assertGreater(abs(get_angle_in_180_range(
                calculate_dihedral_angle(xyz_guess, [n, ca, cb, hb], index=0))), 150.0)

    def test_heuristics_for_qa_hofmann(self):
        """
        Test heuristics for QA Hofmann elimination: CC[N+](C)(C)C + [OH-].O <=> C=C + CN(C)C + O.O.
        """
        rxn = ARCReaction(r_species=[self.ethyl_tma, self.oh_d1],
                          p_species=[self.ethene, self.trimethylamine, self.water_dimer])
        self.assertEqual(rxn.family, 'qa_hofmann')
        adapter = HeuristicsAdapter(job_type='tsg',
                                    reactions=[rxn],
                                    testing=True,
                                    project='test',
                                    project_directory=os.path.join(self.scratch_dir, 'heuristics_qahofmann'))
        with mock.patch.dict(settings, {'UMA_TS_GUESS_PREFILTER': False}):
            adapter.execute_incore()
        self.assertEqual(rxn.family, 'qa_hofmann')
        self.assertTrue(rxn.ts_species.is_ts)
        self.assertGreater(len(rxn.ts_species.ts_guesses), 0)
        for tsg in rxn.ts_species.ts_guesses:
            self.assertEqual(tsg.family, 'qa_hofmann')
        self.check_hofmann_guesses([tsg.initial_xyz for tsg in rxn.ts_species.ts_guesses],
                                   self.ethyl_tma, ca=1, n=2, cb=0, hb_options=[6, 7, 8])

    def test_qa_hofmann_anti_periplanar(self):
        """
        Test QA Hofmann elimination guesses from a conformer with the beta hydrogens gauche to N:
        CCC[N+](C)(C)C + [OH-].O <=> C=CC + CN(C)C + O.O.
        """
        rxn = ARCReaction(r_species=[self.propyl_tma, self.oh_d1],
                          p_species=[self.propene, self.trimethylamine, self.water_dimer])
        with mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                        side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
            xyz_guesses, families = qa_hofmann(rxn)
        self.assertEqual(set(families), {'qa_hofmann'})
        self.check_hofmann_guesses(xyz_guesses, self.propyl_tma, ca=2, n=3, cb=1, hb_options=[10, 11])

    def test_qa_hofmann_benzylic(self):
        """
        Test QA Hofmann elimination guesses for an aryl-activated beta hydrogen:
        C[N+](C)(C)CCc1ccccc1 + [OH-].O <=> C=Cc1ccccc1 + CN(C)C + O.O.
        """
        rxn = ARCReaction(r_species=[self.phenethyl_tma, self.oh_d1],
                          p_species=[self.styrene, self.trimethylamine, self.water_dimer])
        with mock.patch('arc.job.adapters.ts.heuristics_qa.filter_ts_guesses_with_uma_sp',
                        side_effect=lambda xyzs, families, **kwargs: (xyzs, families)):
            xyz_guesses, _ = qa_hofmann(rxn)
        self.check_hofmann_guesses(xyz_guesses, self.phenethyl_tma, ca=4, n=1, cb=5, hb_options=[23, 24],
                                   pi_activated=True)
        # The C-N bond is not stretched for an aryl-activated beta hydrogen.
        for xyz_guess in xyz_guesses:
            self.assertAlmostEqual(calculate_distance(xyz_guess['coords'], [4, 1], index=0),
                                   calculate_distance(self.phenethyl_tma.get_xyz()['coords'], [4, 1], index=0),
                                   delta=0.05)

    @staticmethod
    def _uma_proc(returncode=0, stdout='', stderr=''):
        """Build a stand-in for the ``subprocess.run`` CompletedProcess result."""
        m = mock.Mock()
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = stderr
        return m

    def _uma_guesses(self, n):
        """``n`` TS-guess xyz dicts of distinct geometries (an O-H bond stretched in steps) + their family labels."""
        xyzs = list()
        for i in range(n):
            xyz = copy.deepcopy(self.h2o_xyz)
            coords = [list(coord) for coord in xyz['coords']]
            coords[1][0] += 0.1 * i
            xyz['coords'] = tuple(tuple(coord) for coord in coords)
            xyzs.append(xyz)
        families = [f'qa_sn2_{i}' for i in range(n)]
        return xyzs, families

    def test_remove_duplicate_ts_guesses(self):
        """Test that only TS guesses of the same geometry, up to atom labels, are removed."""
        xyzs, families = self._uma_guesses(3)
        relabeled = {'symbols': ('O', 'H', 'H'), 'isotopes': (16, 1, 1),
                     'coords': (xyzs[1]['coords'][0], xyzs[1]['coords'][2], xyzs[1]['coords'][1])}
        out_xyzs, out_families = remove_duplicate_ts_guesses(xyzs + [copy.deepcopy(xyzs[0]), relabeled],
                                                             families + ['dup_0', 'dup_1'])
        self.assertEqual(out_xyzs, xyzs)
        self.assertEqual(out_families, families)
        # A mirror image has the same energy and leads to the mirror-image TS, so it is also removed.
        mirror = {'symbols': xyzs[2]['symbols'], 'isotopes': xyzs[2]['isotopes'],
                  'coords': tuple((-x, y, z) for x, y, z in xyzs[2]['coords'])}
        out_xyzs, _ = remove_duplicate_ts_guesses(xyzs + [mirror], families + ['mirror'])
        self.assertEqual(out_xyzs, xyzs)
        # Guesses that differ by more than the tolerance are all kept.
        out_xyzs, _ = remove_duplicate_ts_guesses(xyzs, families, tol=0.05)
        self.assertEqual(len(out_xyzs), 3)

    @mock.patch.dict(settings, {'UMA_TS_GUESS_PREFILTER': True})
    def test_filter_ts_guesses_with_uma_sp(self):
        """Test filter_ts_guesses_with_uma_sp`` behaviors, one ``subTest`` per case."""
        mod = filter_ts_guesses_with_uma_sp.__module__

        # No-op cases: <= top_k guesses, subprocess must never be touched.
        for case, n in (('no_cut_below_top_k', 4), ('no_cut_exactly_top_k', 5)):
            with self.subTest(case=case):
                xyzs, families = self._uma_guesses(n)
                with mock.patch.dict(settings, {'UMA_PYTHON': '/fake/uma/python'}), \
                        mock.patch(f'{mod}.subprocess.run') as run:
                    out_xyzs, out_families = filter_ts_guesses_with_uma_sp(xyzs, families)
                run.assert_not_called()
                self.assertEqual(out_xyzs, xyzs)
                self.assertEqual(out_families, families)

        # The pre-filter is off by default -> forward all distinct guesses with a warning, never shell out.
        with self.subTest(case='off_by_default'):
            xyzs, families = self._uma_guesses(8)
            with mock.patch.dict(settings, {'UMA_TS_GUESS_PREFILTER': False}), \
                    mock.patch.dict(settings, {'UMA_PYTHON': '/fake/uma/python'}), \
                    mock.patch(f'{mod}.subprocess.run') as run, \
                    self.assertLogs('arc', level='WARNING') as logs:
                out_xyzs, out_families = filter_ts_guesses_with_uma_sp(xyzs, families)
            run.assert_not_called()
            self.assertEqual(out_xyzs, xyzs)
            self.assertEqual(out_families, families)
            self.assertTrue(any('UMA SP pre-filter is off' in line for line in logs.output))

        # No uma_env (UMA_PYTHON is None) -> forward all distinct guesses with a warning, never shell out.
        with self.subTest(case='fail_safe_uma_absent'):
            xyzs, families = self._uma_guesses(8)
            with mock.patch.dict(settings, {'UMA_PYTHON': None}), \
                    mock.patch(f'{mod}.subprocess.run') as run, \
                    self.assertLogs('arc', level='WARNING') as logs:
                out_xyzs, out_families = filter_ts_guesses_with_uma_sp(xyzs + [copy.deepcopy(xyzs[0])],
                                                                       families + ['duplicate'])
            run.assert_not_called()
            self.assertEqual(out_xyzs, xyzs)
            self.assertEqual(out_families, families)
            self.assertTrue(any('Forwarding all 8 TS guesses unranked' in line for line in logs.output))

        # Worker present but its result is unusable -> forward all.
        partial = json.dumps({f'guess_{i}': float(i) for i in range(5)})  # missing 5,6,7
        for case, run_kwargs in (
            ('fail_safe_nonzero_exit',
             {'return_value': self._uma_proc(returncode=1, stderr='boom')}),
            ('fail_safe_partial_scoring',
             {'return_value': self._uma_proc(stdout=partial)}),
            ('fail_safe_bad_json',
             {'return_value': self._uma_proc(stdout='not json at all')}),
            ('fail_safe_subprocess_exception',
             {'side_effect': subprocess.TimeoutExpired(cmd='uma', timeout=1)}),
        ):
            with self.subTest(case=case):
                xyzs, families = self._uma_guesses(8)
                with mock.patch.dict(settings, {'UMA_PYTHON': '/fake/uma/python'}), \
                        mock.patch(f'{mod}.subprocess.run', **run_kwargs):
                    out_xyzs, out_families = filter_ts_guesses_with_uma_sp(xyzs, families)
                self.assertEqual(out_xyzs, xyzs)
                self.assertEqual(out_families, families)

        # Keep the top_k lowest-energy guesses, in original index order.
        with self.subTest(case='keeps_top_k_lowest_energy_in_original_order'):
            xyzs, families = self._uma_guesses(8)
            # Lowest 5 energies are indices 0,2,3,5,6; indices 1, 4, and 7 are the three highest.
            energies = {'guess_0': 0.0, 'guess_1': 9.0, 'guess_2': 1.0, 'guess_3': 2.0,
                        'guess_4': 8.0, 'guess_5': 3.0, 'guess_6': 4.0, 'guess_7': 5.0}
            with mock.patch.dict(settings, {'UMA_PYTHON': '/fake/uma/python'}), \
                    mock.patch(f'{mod}.subprocess.run',
                               return_value=self._uma_proc(stdout=json.dumps(energies))) as run:
                out_xyzs, out_families = filter_ts_guesses_with_uma_sp(
                    xyzs, families, charge=0, multiplicity=1)
            run.assert_called_once()
            kept = [0, 2, 3, 5, 6]
            self.assertEqual(out_xyzs, [xyzs[i] for i in kept])
            self.assertEqual(out_families, [families[i] for i in kept])

        # Charge/multiplicity are passed through to the worker argv (UMA omol conditioning).
        with self.subTest(case='worker_invoked_with_charge_and_multiplicity'):
            xyzs, families = self._uma_guesses(8)
            energies = json.dumps({f'guess_{i}': float(i) for i in range(8)})
            with mock.patch.dict(settings, {'UMA_PYTHON': '/fake/uma/python'}), \
                    mock.patch(f'{mod}.subprocess.run',
                               return_value=self._uma_proc(stdout=energies)) as run:
                filter_ts_guesses_with_uma_sp(xyzs, families, charge=1, multiplicity=2)
            argv = run.call_args[0][0]
            self.assertEqual(argv[0], '/fake/uma/python')
            self.assertIn('1', argv)  # charge
            self.assertIn('2', argv)  # multiplicity


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
