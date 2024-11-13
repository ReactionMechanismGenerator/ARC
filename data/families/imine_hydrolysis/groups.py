name = "imine_hydrolysis/groups"
shortDesc = u"imine_hydrolysis"
longDesc = u"""
A generic bimolecular imine hydrolysis reaction: R1-A(R2)=N-R3 + H2O <=> R1-A(R2)(OH)-N(H)-R3 

Where:
- A can be C (carbonyl) or P (phosphoryl).
- R1,R2, and R3 represent substituent groups or hydrogen depending on the structure of the ester.

R1-A(*1)(R2)=N(*2)-R3 + H(*3)O(*4)H <=> R1-A(*1)(R2)(O(*4)H)-N(*2)(H(*3))-R3

"""

template(reactants=["imine", "H2O"], products=["acid"], ownReverse=False)

#reverse = "condensation"

#reversible = True

recipe(actions=[
    ['CHANGE_BOND', '*1', -1, '*2'],
    ['BREAK_BOND', '*3', 1, '*4'],
    ['FORM_BOND', '*1', 1, '*4'],
    ['FORM_BOND', '*2', 1, '*3'],
])

entry(
    index = 0,
    label = "imine",
    group = 
"""
1 *1 [C,P]    u0   {2,D}
2 *2 N     u0   {1,D} {3,S}
3    R     u0   {2,S}
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
L1: imine
L1: H2O
"""
)
