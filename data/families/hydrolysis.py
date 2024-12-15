name = "hydrolysis/groups"
shortDesc = u"hydrolysis"
longDesc = u"""
A generic bimolecular hydrolysis reaction: AB + H2O <=> AH + BOH

R1(*1)[O/N](*3)R2(*2) + H(*4)O(*5)H <=> R1(*1)[O/N](*3)H(*4) + R2(*2)O(*5)H

"""

template(reactants=["R1ONR2", "H2O"], products=["R1ONH", "R2OH"], ownReverse=False)

reverse = "condensation"

reversible = True

recipe(actions=[
    ['BREAK_BOND', '*3', 1, '*2'],
    ['BREAK_BOND', '*4', 1, '*5'],
    ['FORM_BOND', '*3', 1, '*4'],
    ['FORM_BOND', '*2', 1, '*5'],
])

entry(
    index = 0,
    label = "R1ONR2",
    group = 
"""
1 *1 R!H       u0 p0     c0 {2,S}
2 *3 [O2s,N3s] u0 p[1,2] c0 {1,S} {3,S}
3 *2 R!H       u0 p0     c0 {2,S}
""",
    kinetics = None,
)

entry(
    index = 1,
    label = "H2O",
    group = 
"""
1 *4 O u0 p2 c0 {2,S} {3,S}
2 *5 H u0 p0 c0 {1,S}
3    H u0 p0 c0 {1,S}
""",
    kinetics = None,
)


tree(
"""
L1: R1ONR2
L1: H2O
"""
)

