#!/usr/bin/env python
# encoding: utf-8

name = "qa_sn2"
shortDesc = u""
longDesc = u"""
This family describes SN2 degradation reactions of quaternary ammonium species with hydroxide,
where the hydroxide is represented together with one explicit water molecule:

R3C(N+)R3 + (OH-·H2O) -> R3COH·H2O + NR3

The hydroxide oxygen attacks the carbon adjacent to the quaternary ammonium nitrogen.
At the same time, the C-N bond breaks, and the quaternary ammonium group is eliminated
as a neutral tertiary amine.

Atom labeling:
*1 - reacting carbon adjacent to the quaternary ammonium group
*2 - nitrogen of the quaternary ammonium leaving group
*3 - hydroxide oxygen that forms the new C-O bond
"""

template(reactants=["R3CNR3", "OH_d1"], products=["R3COH_d1", "NR3"], ownReverse=False)

reversible = True

reactantNum = 2

productNum = 2

allowChargedSpecies = True



recipe(actions=[
    ['BREAK_BOND', '*1', 1, '*2'],
    ['FORM_BOND', '*1', 1, '*3'],
    ['LOSE_PAIR', '*3', '1'],
    ['GAIN_PAIR', '*2', '1'],

])

entry(
    index = 0,
    label = "R3CNR3",
    group =
"""
1 *1 C u0 p0 c0 {2,S} {6,S} {7,S} {8,S}
2 *2 N u0 p0 c+1 {1,S} {3,S} {4,S} {5,S}
3    R u0 p0 c0 {2,S}
4    R u0 p0 c0 {2,S}
5    R u0 p0 c0 {2,S}
6    R u0 p0 c0 {1,S}
7    R u0 p0 c0 {1,S}
8    R u0 p0 c0 {1,S}
""",
    kinetics = None,
)

entry(
    index = 1,
    label = "OH_d1",
    group =
"""
1 *3 O u0 p3 c-1 {2,S}
2    H u0 p0 c0 {1,S}
3    O u0 p2 c0 {4,S} {5,S}
4    H u0 p0 c0 {3,S}
5    H u0 p0 c0 {3,S}
""",
    kinetics = None,
)

entry(
    index = 2,
    label = "R3COH_d1",
    group =
"""
1 *1 C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
2 *3 O u0 p2 c0 {1,S} {3,S}
3    H u0 p0 c0 {2,S}
4    R u0 p0 c0 {1,S}
5    R u0 p0 c0 {1,S}
6    R u0 p0 c0 {1,S}
7    O u0 p2 c0 {8,S} {9,S}
8    H u0 p0 c0 {7,S}
9    H u0 p0 c0 {7,S}
""",
    kinetics = None,
)

entry(
    index = 3,
    label = "NR3",
    group =
"""
1 *2 N u0 p1 c0 {2,S} {3,S} {4,S}
2    R u0 p0 c0 {1,S}
3    R u0 p0 c0 {1,S}
4    R u0 p0 c0 {1,S}
""",
    kinetics = None,
)


tree(
"""
L1: R3CNR3
L1: OH_d1
"""
)


