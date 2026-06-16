#!/usr/bin/env python
# encoding: utf-8

name = "qa_hofmann_elimination"
shortDesc = u""
longDesc = u"""
This family describes Hofmann elimination reactions of quaternary ammonium species
with hydroxide, where the hydroxide is represented together with one explicit water molecule:

R2C(H)-C(R2)-[N+](R3) + (OH−·H2O) → R2C=C(R2) + H2O·H2O + NR3

The hydroxide oxygen abstracts a beta hydrogen from the carbon adjacent to the alpha carbon.
At the same time, the C_alpha-N bond breaks and a C_alpha=C_beta double bond is formed.
The quaternary ammonium group is eliminated as a neutral tertiary amine.

Atom labeling:
*1 - alpha carbon bonded to the quaternary ammonium nitrogen
*2 - nitrogen of the quaternary ammonium leaving group
*3 - beta carbon bearing the abstracted hydrogen
*4 - beta hydrogen abstracted by hydroxide
*5 - hydroxide oxygen that abstracts the beta hydrogen
"""

template(
    reactants=["R3CNCR3", "OH_d1"],
    products=["alkene", "NR3", "H2O_d1"],
    ownReverse=False,
)

reversible = True

reactantNum = 2

productNum = 3

allowChargedSpecies = True


recipe(actions=[
    ['BREAK_BOND', '*1', 1, '*2'],
    ['BREAK_BOND', '*3', 1, '*4'],
    ['CHANGE_BOND', '*1', 1, '*3'],
    ['FORM_BOND', '*5', 1, '*4'],
    ['LOSE_PAIR', '*5', '1'],
    ['GAIN_PAIR', '*2', '1'],
])


entry(
    index = 0,
    label = "R3CNCR3",
    group =
"""
1 *1 C u0 p0 c0 {2,S} {3,S} {6,S} {7,S}
2 *2 N u0 p0 c+1 {1,S} {8,S} {9,S} {10,S}
3 *3 C u0 p0 c0 {1,S} {4,S} {11,S} {12,S}
4 *4 H u0 p0 c0 {3,S}
6    R u0 p0 c0 {1,S}
7    R u0 p0 c0 {1,S}
8    R u0 p0 c0 {2,S}
9    R u0 p0 c0 {2,S}
10   R u0 p0 c0 {2,S}
11   R u0 p0 c0 {3,S}
12   R u0 p0 c0 {3,S}
""",
    kinetics = None,
)


entry(
    index = 1,
    label = "OH_d1",
    group =
"""
1 *5 O u0 p3 c-1 {2,S}
2    H u0 p0 c0 {1,S}
3    O u0 p2 c0 {4,S} {5,S}
4    H u0 p0 c0 {3,S}
5    H u0 p0 c0 {3,S}
""",
    kinetics = None,
)


entry(
    index = 2,
    label = "alkene",
    group =
"""
1 *1 C u0 p0 c0 {2,D} {3,S} {4,S}
2 *3 C u0 p0 c0 {1,D} {5,S} {6,S}
3    R u0 p0 c0 {1,S}
4    R u0 p0 c0 {1,S}
5    R u0 p0 c0 {2,S}
6    R u0 p0 c0 {2,S}
""",
    kinetics = None,
)


entry(
    index = 3,
    label = "NR3",
    group =
"""
1 *2 N u0 p1 c0 {2,S} {3,S} {4,S}
2    R u0 p0 c0 {1,S}
3    R u0 p0 c0 {1,S}
4    R u0 p0 c0 {1,S}
""",
    kinetics = None,
)


entry(
    index = 4,
    label = "H2O_d1",
    group =
"""
1 *5 O u0 p2 c0 {2,S} {3,S}
2 *4 H u0 p0 c0 {1,S}
3    H u0 p0 c0 {1,S}
4    O u0 p2 c0 {5,S} {6,S}
5    H u0 p0 c0 {4,S}
6    H u0 p0 c0 {4,S}
""",
    kinetics = None,
)


tree(
"""
L1: R3CNCR3
L1: OH_d1
"""
)
