name = "nitrile/groups"
shortDesc = u"nitrile_hydrolysis"
longDesc = u"""
A generic bimolecular ester hydrolysis reaction: R1-A#N + H2O <=> R1-A(OH)=NH 

Where:
- A can be C (carbonyl) or P (phosphoryl).
- R1 represent substituent groups or hydrogen depending on the structure of the ester.

R1-A(*1)#N(*2) + H(*3)O(*4)H <=> R1-A(O(*4)H)=N(*2)H(*3)

"""

template(reactants=["nitrile", "H2O"], products=["acid"], ownReverse=False)

reverse = "condensation"

reversible = True

recipe(actions=[
    ['CHANGE_BOND', '*1', -1, '*2'],
    ['BREAK_BOND', '*3', 1, '*4'],
    ['FORM_BOND', '*1', 1, '*4'],
    ['FORM_BOND', '*2', 1, '*3'],
])

entry(
    index = 0,
    label = "nitrile",
    group = 
"""
1    R    u0 {2,S}
2 *1 [C,P]    u0 {1,S} {3,T}
3 *2 N    u0 {2,T} 
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
L1: nitrile
L1: H2O
"""
)
