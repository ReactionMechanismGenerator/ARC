name = "carbonyl_based_hydrolysis/groups"
shortDesc = u"carbonyl_based_hydrolysis"
longDesc = u"""
A generic bimolecular carbonyl_based hydrolysis reaction: R-A(=O)-B + H2O <=> R-A(=O)-OH + B-H

Where:
- A can be C (carbonyl) or P (phosphoryl).
- B can be O, N, or a halide (X).
- R represents substituent groups or hydrogen depending on the structure of the ester.

RA(*1)(=O)B(*2) + H(*3)O(*4)H <=> RA(*1)(=O)O(*4)H + B(*2)H(*3)


This family encompasses hydrolysis reactions for a range of carbonyl_based classes, including esters, amides, and acyl halides.
"""

template(reactants=["carbonyl_group", "H2O"], products=["acid", "alcohol"], ownReverse=False)

reverse = "condensation"

reversible = True

recipe(actions=[
    ['BREAK_BOND', '*1', 1, '*2'],
    ['BREAK_BOND', '*3', 1, '*4'],
    ['FORM_BOND', '*1', 1, '*4'],
    ['FORM_BOND', '*2', 1, '*3'],
])

entry(
    index = 0,
    label = "carbonyl_group",
    group = 
"""
1    R               u0 {2,S}
2 *1 [C,P]           u0 {1,S} {3,D} {4,S}
3    O               u0 {2,D}
4 *2 [O,N,Cl,Br,F,I] u0 {2,S}
""",
    kinetics = None,
)

entry(
    index = 1,
    label = "H2O",
    group = 
"""
1 *4 O u0 p2 c0 {2,S} {3,S}
2 *3 H u0 p0 c0 {1,S}
3    H u0 p0 c0 {1,S}
""",
    kinetics = None,
)


tree(
"""
L1: carbonyl_group
L1: H2O
"""
)
