"""
This creates atomTypes and assigns them with the correct bond/lone_pairs/charge
Used in isomorphismTest.py to create group_atomtypes
"""


class AbstractAtomType(object):
    def __init__(self, element=None, label=None, double=-1, triple=-1, quadruple=-1, benzene=-1, lp=-1, chrg=-1):
        self.element = element
        self.label = label
        self.double = double
        self.triple = triple
        self.quadruple = quadruple
        self.benzene = benzene
        self.lp = lp
        self.chrg = chrg


class Column4(AbstractAtomType):  # C
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.lp = 0


class Column5(AbstractAtomType):  # N
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.lp = 1


class Column6(AbstractAtomType):  # O, S
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.lp = 2


class Xs(AbstractAtomType):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.double, self.triple, self.benzene, self.quadruple = 0, 0, 0, 0
        self.label = 's'


class Xd(AbstractAtomType):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.double, self.triple, self.benzene, self.quadruple = 1, 0, 0, 0
        self.label = 'd'


class Xdd(AbstractAtomType):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.double, self.triple, self.benzene, self.quadruple = 2, 0, 0, 0
        self.label = 'dd'


class Xt(AbstractAtomType):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.double, self.triple, self.benzene, self.quadruple = 0, 1, 0, 0
        self.label = 't'


class Xq(AbstractAtomType):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.double, self.triple, self.benzene, self.quadruple = 0, 0, 0, 1
        self.label = 'q'


class Xb(AbstractAtomType):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.double, self.triple, self.benzene, self.quadruple = 0, 0, 2, 0
        self.label = 'b'


class Xbf(AbstractAtomType):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.double, self.triple, self.benzene, self.quadruple = 0, 0, 3, 0
        self.label = 'bf'


def create_types(Type, elements, labels=None):
    if labels is None:
        labels = elements
    atomtypes = list()
    for el, label in zip(elements, labels):
        at = Type(element=el)
        at.label = label + at.label
        atomtypes.append(at)
    return atomtypes
