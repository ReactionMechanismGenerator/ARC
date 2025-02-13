name = "ether_hydrolysis/groups"
shortDesc = u"ether_hydrolysis"
longDesc = u"""
A generic bimolecular ether hydrolysis reaction: A-B-R + H2O <=> A-OH + B(R)-H

Where:
- A can be C 
- B can be O
- R represents an alkyl (R'') or aryl (Ar) group.

A(*1)B(*2)-R + H(*3)O(*4)H <=> A(*1)O(*4)H + B(*2)(R)H(*3)

"""

template(reactants=["ether", "H2O"], products=["alcohol1", "alcohol2"], ownReverse=False)

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
    label = "ether",
    group = 
"""
1 *1  C u0 {2,S} 
2 *2  O u0 {1,S} {3,S}
3     C u0 {2,S}
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
L1: ether
L1: H2O
"""
)
