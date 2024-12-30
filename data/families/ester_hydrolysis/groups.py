name = "ester_hydrolysis/groups"
shortDesc = u"ester_hydrolysis"
longDesc = u"""
A generic bimolecular ester hydrolysis reaction: R1-A(=C)-B(R2)-R3 + H2O <=> R1-A(=C)-OH + R3-B(R2)-H

Where:
- A can be C (carbonyl) or P (phosphoryl).
- B can be O, S, N, or a halide (X).
- R1, R2, and R3 represent substituent groups or hydrogen depending on the structure of the ester.

R1A(*1)(=C)B(*2)R2 + H(*3)O(*4)H <=> R1A(*1)(=C)O(*4)H + R2B(*2)H(*3)


This family encompasses hydrolysis reactions for a range of ester types, including carboxylic, phosphoric, and thioester variants.
"""

template(reactants=["ester", "H2O"], products=["acid", "alcohol"], ownReverse=False)

#reverse = "condensation"

#reversible = True

recipe(actions=[
    ['BREAK_BOND', '*1', 1, '*2'],
    ['BREAK_BOND', '*3', 1, '*4'],
    ['FORM_BOND', '*1', 1, '*4'],
    ['FORM_BOND', '*2', 1, '*3'],
])

entry(
    index = 0,
    label = "ester",
    group = 
"""
1    R    u0 {2,S}
2 *1 [C,P]    u0 {1,S} {3,D} {4,S}
3    O    u0 {2,D}
4 *2 [O,N,Cl,Br,F]    u0 {2,S}
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
L1: ester
L1: H2O
"""
)
