#!/usr/bin/env python
# encoding: utf-8

name = "qa_ring_opening"
shortDesc = u""
longDesc = u"""
This family describes the ring-opening SN2 degradation reactions of cyclic quaternary ammonium species with hydroxide,
where the hydroxide is represented together with one explicit water molecule:

cyclo-[R2C-N+(R)] + (OH-·H2O) -> HO-R2C-...-NR2·H2O

The hydroxide oxygen attacks a ring carbon adjacent to the quaternary ammonium nitrogen. At the same time,
the ring C-N bond breaks, and the ring opens into a single amino alcohol product.
This is the QA SN2 reaction (see qa_sn2) with the Ca-N bond in a ring, hence a single product.
Only two of the other N neighbours are listed (the fourth bond of N+ follows from its valence), so the
groups also match a three-membered ring (aziridinium), where Ca and N share their ring neighbour.

Atom labeling:
*1 - Reacting ring carbon adjacent to the quaternary ammonium group
*2 - Nitrogen of the quaternary ammonium leaving group
*3 - Hydroxide oxygen that forms the new C-O bond
"""

template(reactants=["R3CNR3", "OH_d1"], products=["R3COH_NR2_d1"], ownReverse=False)

reversible = True

reactantNum = 2

productNum = 1

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
1 *1 C u0 p0 c0 {2,S} {5,S} {6,S} {7,S}
2 *2 N u0 p0 c+1 {1,S} {3,S} {4,S}
3    R u0 p0 c0 {2,S}
4    R u0 p0 c0 {2,S}
5    R u0 p0 c0 {1,S}
6    R u0 p0 c0 {1,S}
7    R u0 p0 c0 {1,S}
""",
    kinetics = None,
)

entry(
    index = 1,
    label = "OH_d1",
    group =
"""
1 *3 O u0 p3 c-1 {2,S}
2    H u0 p0 c0 {1,S}
3    O u0 p2 c0 {4,S} {5,S}
4    H u0 p0 c0 {3,S}
5    H u0 p0 c0 {3,S}
""",
    kinetics = None,
)

entry(
    index = 2,
    label = "R3COH_NR2_d1",
    group =
"""
1 *1 C u0 p0 c0 {2,S} {4,S} {5,S} {6,S}
2 *3 O u0 p2 c0 {1,S} {3,S}
3    H u0 p0 c0 {2,S}
4    R u0 p0 c0 {1,S}
5    R u0 p0 c0 {1,S}
6    R u0 p0 c0 {1,S}
7 *2 N u0 p1 c0 {8,S} {9,S}
8    R u0 p0 c0 {7,S}
9    R u0 p0 c0 {7,S}
10   O u0 p2 c0 {11,S} {12,S}
11   H u0 p0 c0 {10,S}
12   H u0 p0 c0 {10,S}
""",
    kinetics = None,
)


tree(
"""
L1: R3CNR3
L1: OH_d1
"""
)
