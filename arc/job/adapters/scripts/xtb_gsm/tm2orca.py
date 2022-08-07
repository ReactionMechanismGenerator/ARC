#!/usr/bin/env python
# encoding: utf-8
'''
Created on 26.06.2017
Fixed on 01.06.2020 by Sebastian Ehlert

@author: dohm
'''

from __future__ import print_function

import sys, os
element_symbols_lc=["bq","h","he","li","be","b","c","n","o","f","ne",
                 "na","mg","al","si","p","s","cl","ar","k","ca","sc",
                 "ti","v","cr","mn","fe","co","ni","cu","zn","ga","ge",
                 "as","se","br","kr","rb","sr","y","zr","nb","mo","tc",
                 "ru","rh","pd","ag","cd","in","sn","sb","te","i","xe",
                 "cs","ba","la","ce","pr","nd","pm","sm","eu","gd","tb",
                 "dy","ho","er","tm","yb","lu","hf","ta","w","re","os",
                 "ir","pt","au","hg","tl","pb","bi","po","at","rn","fr",
                 "ra","ac","th","pa","u","np","pu","am","cm","bk","cf",
                 "es","fm","md","no","lr","rf","db","sg","bh","hs","mt",
                 "ds","rg","cn","ut","fl","up","lv","us","uo"]

#element_symbols_lc = map(lambda x:x.lower(),element_symbols)

argcount = len(sys.argv)
if (argcount != 2):
    print(u"need Basename, exiting...")
    quit()
#assign cli to strings
basename = str(sys.argv[1])

engrad = open(basename + ".engrad", 'w')
gradient = open('gradient', 'r')
energy = open('energy', 'r')
outfile = open(basename + ".out", 'w')

enline = energy.readlines()[-2]
energy_value = float(enline.strip().split()[1])
gradlines = gradient.readlines()
natoms = int((len(gradlines) - 3) / 2)
coordlines = gradlines[2:2+natoms]
gradvallines= gradlines[2+natoms:2+2*natoms]

#write engrad file:
engrad.write(u"#\n# Number of atoms\n#\n")
engrad.write(u"{:3d}".format(natoms) +u"\n")
engrad.write(u"#\n# The current total energy in Eh\n#"+u"\n")
engrad.write(u'{:20.12f}'.format(energy_value)+u"\n")
engrad.write(u"#\n# The current gradient in Eh/bohr\n#"+u"\n")
for gl in gradvallines:
    glval = gl.strip().split()
    for gv in glval:
        gv = gv.replace("D","E")
        engrad.write(u'{:21.12f}'.format(float(gv))+u"\n")
engrad.write(u"#\n# The atomic numbers and current coordinates in Bohr\n#"+u"\n")
for cl in coordlines:
    cl = cl.replace("D","E")
    
    cls = cl.strip().split()
    coords = [float(cls[0]),float(cls[1]),float(cls[2])]
    atn = element_symbols_lc.index(cls[3].lower())
    engrad.write(u'{:4d} {:13.7f}{:13.7f}{:13.7f}'.format(atn, coords[0],coords[1],coords[2])+u"\n")
engrad.close()

#write .out file:

outfile.write(u"ORCA-Dummy output for GSM TS Optimizer. Not a real ORCA-Output"+u"\n")
outfile.write(u"Total Energy       :  {:20.8f}".format(energy_value)+u"\n")
outfile.write(u"------------------\nCARTESIAN GRADIENT\n------------------\n\n")

for i in range(natoms):
    gl = gradvallines[i].replace("D","E")
    cl = coordlines[i].replace("D","E")    
    cls = cl.strip().split()
    gls = gl.strip().split()    
    grads = [float(gls[0]),float(gls[1]),float(gls[2])]    
    outfile.write(u'{:4d}{:>4}   :{:15.9f}{:15.9f}{:15.9f}'.format(i, cls[3],grads[0],grads[1],grads[2])+u"\n")

gradient.close()
energy.close()

os.rename(u"energy", basename+u".energy")
os.rename(u"gradient", basename+u".gradient")

